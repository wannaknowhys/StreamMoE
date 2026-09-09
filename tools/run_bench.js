// run_bench.js - placement throughput benchmark runner.
//
// Reuses the run_specs model/engine/task cartesian structure from run_export.js.
// Placement is expressed in the engine spec `extra` args:
//   --dense-placement C1:<dev>,C2:<dev>          (RAM | CPU | Vulkan0)
//   --moe-expert-pools RAM:<MB>[,Vulkan0:<MB>]
// A stock upstream binary is driven through engine `binPath`.
//
// Task kinds:
//   single  : { prompt | promptFile, nPredict, warmup, repeat, ctx, threads }
//             cold -> warmup (discarded) -> repeat (median); /completion with
//             cache_prompt=false so every repeat re-prefills the whole prompt.
//   jsonl   : { feed:{type:"jsonl", path, maxTurns}, nPredict, ctx, threads }
//             chained multi-turn; /v1/chat/completions with cache_prompt=true.
//             Every turn is one JSONL record + one console line; a final
//             summary record carries the aggregate and the server log tail.
//
// Usage:
//   node tools/run_bench.js --models <spec[,...]> --engines <spec[,...]> --tasks <spec[,...]>
//
// Env:
//   SM_BENCH_PORT       server port (default 8994)
//   SM_BENCH_OUT        result jsonl (default benchmark/results/bench_<ts>.jsonl)
//   ${VAR} in spec files expands from env (machine paths stay out of specs)
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.join(__dirname, '..');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function parseArgv() {
    const a = process.argv.slice(2);
    const p = {};
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '--models') p.models = a[++i];
        else if (a[i] === '--engines') p.engines = a[++i];
        else if (a[i] === '--tasks') p.tasks = a[++i];
        else if (a[i] === '--port') p.port = Number(a[++i]);
        else if (a[i] === '--health-timeout') p.healthTimeout = Number(a[++i]) * 1000;
        else if (a[i] === '--out') p.out = a[++i];
    }
    return p;
}

function readSpecList(csv) {
    return csv.split(',').map((f) => JSON.parse(fs.readFileSync(f.trim(), 'utf8').replace(/^\uFEFF/, '')));
}

function absPath(p) {
    return path.isAbsolute(p) ? p : path.join(REPO_ROOT, p);
}

// Resolve ${X}: run field wins, then env. Called AFTER merge so ${pool} (a
// model spec field) resolves against the merged run.
function expandRun(run) {
    const expand = (s) => s.replace(/\$\{([A-Za-z0-9_]+)\}/g, (m, k) => {
        if (run[k] !== undefined) return String(run[k]);
        if (process.env[k] !== undefined) return process.env[k];
        throw new Error('[run_bench] unresolvable ${' + k + '}');
    });
    const out = {};
    for (const [k, v] of Object.entries(run)) {
        if (typeof v === 'string') out[k] = expand(v);
        else if (Array.isArray(v)) out[k] = v.map((x) => (typeof x === 'string' ? expand(x) : x));
        else if (v && typeof v === 'object') out[k] = expandRun(v);
        else out[k] = v;
    }
    return out;
}

function cartesian(models, engines, tasks) {
    const out = [];
    for (const m of models) for (const e of engines) for (const t of tasks) {
        const run = {};
        const merge = (src) => {
            for (const [k, v] of Object.entries(src)) {
                if (k in run) throw new Error('[run_bench] duplicate key across specs: ' + k);
                run[k] = v;
            }
        };
        merge(m); merge(e); merge(t);
        out.push(expandRun(run));
    }
    return out;
}

function binPath(run) {
    if (run.binPath) return run.binPath;
    if (!run.bin) throw new Error('[run_bench] run.bin or run.binPath required');
    return path.join(REPO_ROOT, 'build', run.bin, 'llama-build', 'bin', 'llama-server.exe');
}

// Inline prompt, or the first line's prompt/content of a prompt file.
function loadPrompt(run) {
    let base;
    if (run.prompt) {
        base = run.prompt;
    } else if (run.promptFile) {
        const raw = fs.readFileSync(absPath(run.promptFile), 'utf8').replace(/^\uFEFF/, '');
        const first = raw.trim().split('\n')[0];
        let obj = null;
        try { obj = JSON.parse(first); } catch (_) {}
        base = obj ? (obj.prompt || obj.content || first) : first;
    } else {
        throw new Error('[run_bench] task needs prompt or promptFile');
    }
    const n = Number(run.promptRepeat || 1);
    return n > 1 ? new Array(n).fill(base).join('\n\n') : base;
}

// Deterministic filler sized to ~targetTokens. CHARS_PER_TOKEN is a rough
// English-prose estimate; the exact count is reported per request as prompt_n,
// so every config sees the same prompt regardless of tokenizer.
const CHARS_PER_TOKEN = 5.5;
function makeFillerPrompt(targetTokens) {
    const base = 'The memory-bounded mixture-of-experts engine streams expert weights from disk while keeping dense attention layers fully resident.';
    const perRepeat = Math.max(1, Math.round(base.length / CHARS_PER_TOKEN));
    const n = Math.max(1, Math.ceil(targetTokens / perRepeat));
    return new Array(n).fill(base).join(' ');
}

// Rough token count (CJK-aware) and a character slice that targets it. Only used
// to size/trim a prefill prompt; the exact count is reported as prompt_n.
function estTokens(s) {
    const cjk = (s.match(/[\u4e00-\u9fff]/g) || []).length;
    return cjk / 1.2 + (s.length - cjk) / 4;
}
// Character prefix of ~target*4 (rough English tokens); exact count is prompt_n.
function sliceToTokens(s, target) {
    const budget = target * 4;
    return s.length <= budget ? s : s.slice(0, budget);
}

// A prefill snapshot is a JSON array of chat messages. Flatten it to plain text
// so /completion works for any model (tool-call templates reject non-tool models).
function flattenMessages(msgs) {
    return msgs.map((m) => {
        const s = typeof m.content === 'string' ? m.content : JSON.stringify(m.content || '');
        return '[' + (m.role || '?') + ']\n' + s;
    }).join('\n\n');
}

function loadPrefillPrompt(pathStr) {
    const raw = fs.readFileSync(absPath(pathStr), 'utf8').replace(/^\uFEFF/, '');
    const msgs = JSON.parse(raw);
    if (!Array.isArray(msgs)) throw new Error('[run_bench] prefill path must be a JSON array of chat messages');
    return flattenMessages(msgs);
}

function loadFeed(run) {
    if (!run.feed) return { mode: 'single', prompt: loadPrompt(run) };
    const f = run.feed;
    if (f.type === 'jsonl') {
        const raw = fs.readFileSync(absPath(f.path), 'utf8').replace(/^\uFEFF/, '');
        const lines = raw.trim().split('\n').map((l) => JSON.parse(l));
        return { mode: 'jsonl', lines, maxTurns: Number(f.maxTurns || lines.length) };
    }
    if (f.type === 'prefill') {
        if (f.path) {
            let prompt = loadPrefillPrompt(f.path);
            if (f.tokens) prompt = sliceToTokens(prompt, Number(f.tokens));
            return { mode: 'single', prefill: true, prompt, tokens: Math.round(estTokens(prompt)) };
        }
        const tokens = Number(f.tokens || 10000);
        return { mode: 'single', prefill: true, tokens, prompt: makeFillerPrompt(tokens) };
    }
    throw new Error('[run_bench] unsupported feed.type: ' + f.type);
}

// Reject on spawn error or early exit so a missing binary / OOM load does not
// burn the whole health timeout.
function waitHealthy(child, port, timeoutMs) {
    return new Promise((resolve, reject) => {
        const t0 = Date.now();
        let done = false;
        const finish = (err) => {
            if (done) return;
            done = true;
            child.removeListener('error', onErr);
            child.removeListener('exit', onExit);
            err ? reject(err) : resolve();
        };
        const onErr = (e) => finish(new Error('spawn error: ' + e.message));
        const onExit = (code) => finish(new Error('server exited during load, code=' + code));
        child.once('error', onErr);
        child.once('exit', onExit);
        const poll = () => {
            if (done) return;
            if (Date.now() - t0 > timeoutMs) return finish(new Error('health timeout after ' + Math.round(timeoutMs / 1000) + 's'));
            const req = http.get(`http://127.0.0.1:${port}/health`, (res) => {
                res.resume();
                if (res.statusCode === 200) finish();
                else setTimeout(poll, 1000);
            });
            req.on('error', () => setTimeout(poll, 1000));
        };
        poll();
    });
}

function postJson(port, urlPath, payload) {
    return new Promise((resolve, reject) => {
        const body = JSON.stringify(payload);
        const req = http.request({ hostname: '127.0.0.1', port, path: urlPath, method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (res) => {
            let d = '';
            res.on('data', (c) => (d += c));
            res.on('end', () => {
                try { resolve(JSON.parse(d)); } catch (e) { reject(new Error('bad JSON from ' + urlPath + ': ' + d.slice(0, 200))); }
            });
        });
        req.on('error', reject);
        req.end(body);
    });
}

function postShutdown(port) {
    return postJson(port, '/shutdown', {}).catch(() => {});
}

function statsOf(t) {
    return {
        prompt_n: t.prompt_n, prompt_ms: t.prompt_ms, prompt_tps: t.prompt_per_second,
        gen_n: t.predicted_n, gen_ms: t.predicted_ms, decode_tps: t.predicted_per_second,
        cache_n: t.cache_n,
    };
}

// Single-shot /completion. cache_prompt=false: every call re-prefills.
async function postCompletion(port, prompt, nPredict) {
    const j = await postJson(port, '/completion', { prompt, n_predict: nPredict, temperature: 0, cache_prompt: false, stream: false });
    if (j.error) return { error: String(j.error.message || j.error) };
    return statsOf(j.timings || {});
}

// Chained /v1/chat/completions. cache_prompt=true: real multi-turn reuse.
async function postChat(port, messages, nPredict) {
    const j = await postJson(port, '/v1/chat/completions', { model: 'm', messages, max_tokens: nPredict, temperature: 0, cache_prompt: true, stream: false });
    if (j.error) return { error: String(j.error.message || j.error) };
    const msg = j.choices && j.choices[0] && j.choices[0].message;
    return Object.assign(statsOf(j.timings || {}), { content: msg ? (msg.content || '') : '' });
}

const mean = (xs) => { const a = xs.filter((x) => typeof x === 'number' && isFinite(x)); return a.length ? a.reduce((s, x) => s + x, 0) / a.length : null; };
function median(xs) {
    const a = xs.filter((x) => typeof x === 'number' && isFinite(x)).sort((x, y) => x - y);
    if (!a.length) return null;
    const m = Math.floor(a.length / 2);
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
const fmt = (x, d) => (x == null ? '-' : Number(x).toFixed(d == null ? 2 : d));

async function runOne(run, port, healthTimeout) {
    const bin = binPath(run);
    const threads = Number(run.threads || 16);
    const nPredict = Number(run.nPredict || 128);
    const warmup = run.warmup != null ? Number(run.warmup) : 1;
    const repeat = Number(run.repeat || 3);

    let feed;
    try {
        feed = loadFeed(run);
    } catch (e) {
        const rec = { ts: new Date().toISOString(), model: run.model, engine: run.engine, input: run.input, bin, kind: 'single', status: 'FAIL', error: e.message };
        console.error('  FAILED: ' + e.message);
        return { records: [rec], summary: { model: run.model, engine: run.engine, input: run.input, status: 'FAIL', error: e.message } };
    }

    // prefill feed sizes ctx to fit the prompt (+ decode + margin) unless set.
    const ctx = Number(run.ctx || (feed.prefill ? Math.max(16384, (feed.tokens || 0) + nPredict + 4096) : 8192));
    const args = ['-m', run.modelPath, '--host', '127.0.0.1', '--port', String(port),
        '-c', String(ctx), '-t', String(threads), '--no-warmup', '--no-webui'];
    if (run.draft) args.push('--model-draft', run.draft);
    if (run.extra) args.push(...run.extra);

    const base = { ts: new Date().toISOString(), model: run.model, engine: run.engine, input: run.input, bin, args };
    const records = [];
    let summary;

    const mode = feed.prefill ? 'prefill' : feed.mode;
    console.log(`\n=== ${path.basename(bin)} ${run.model}/${run.engine}/${run.input} (${mode}) ===`);
    const child = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let log = '';
    let exited = false;
    child.stdout.on('data', (d) => (log += d));
    child.stderr.on('data', (d) => (log += d));
    child.on('exit', () => { exited = true; });

    try {
        await waitHealthy(child, port, healthTimeout);

        if (feed.mode === 'single') {
            const rec = Object.assign({}, base, { kind: 'single', status: 'OK', error: null, cold: null, warmup: [], steady: [], median: null });
            rec.cold = await postCompletion(port, feed.prompt, nPredict);
            if (rec.cold.error) throw new Error('cold request: ' + rec.cold.error);
            for (let i = 0; i < warmup; i++) {
                const r = await postCompletion(port, feed.prompt, nPredict);
                if (r.error) throw new Error('warmup request: ' + r.error);
                rec.warmup.push(r);
            }
            for (let i = 0; i < repeat; i++) {
                const r = await postCompletion(port, feed.prompt, nPredict);
                if (r.error) throw new Error('steady request: ' + r.error);
                rec.steady.push(r);
            }
            rec.median = {
                prompt_n: rec.steady[0] ? rec.steady[0].prompt_n : null,
                prompt_tps: median(rec.steady.map((r) => r.prompt_tps)),
                decode_tps: median(rec.steady.map((r) => r.decode_tps)),
            };
            records.push(rec);
            console.log(`  cold tg=${fmt(rec.cold.decode_tps)} | warm tg=${fmt(rec.median.decode_tps)} pp=${fmt(rec.median.prompt_tps, 1)} (n=${repeat})`);
            summary = { model: run.model, engine: run.engine, input: run.input, status: 'OK', decode_tps: rec.median.decode_tps, prompt_tps: rec.median.prompt_tps, detail: 'n=' + repeat };
        } else {
            const messages = [];
            const turnTps = [];
            const ppTps = [];
            let totalPrompt = 0, totalGen = 0;
            for (let i = 0; i < feed.maxTurns; i++) {
                const { system, prompt } = feed.lines[i];
                if (system && String(system).trim()) messages.push({ role: 'system', content: system });
                messages.push({ role: 'user', content: prompt });
                const r = await postChat(port, messages, nPredict);
                if (r.error) throw new Error('turn ' + (i + 1) + ': ' + r.error);
                messages.push({ role: 'assistant', content: r.content });
                records.push(Object.assign({}, base, {
                    kind: 'turn', status: 'OK', turn: i + 1,
                    prompt_tokens: r.prompt_n, prompt_tps: r.prompt_tps, decode_tps: r.decode_tps, gen_n: r.gen_n, cache_n: r.cache_n,
                }));
                turnTps.push(r.decode_tps);
                ppTps.push(r.prompt_tps);
                totalPrompt += r.prompt_n || 0;
                totalGen += r.gen_n || 0;
                console.log(`  turn ${i + 1}/${feed.maxTurns}  prompt=${r.prompt_n}  pp=${fmt(r.prompt_tps, 1)}  tg=${fmt(r.decode_tps)}`);
            }
            const agg = {
                turns: records.filter((r) => r.kind === 'turn').length,
                total_prompt_tokens: totalPrompt, total_gen_tokens: totalGen,
                avg_decode_tps: mean(turnTps), median_decode_tps: median(turnTps),
                avg_prompt_tps: mean(ppTps),
            };
            records.push(Object.assign({}, base, { kind: 'summary', status: 'OK' }, agg));
            summary = Object.assign({ model: run.model, engine: run.engine, input: run.input, status: 'OK', decode_tps: agg.avg_decode_tps, prompt_tps: agg.avg_prompt_tps }, agg);
            console.log(`  == ${agg.turns} turns: avg tg=${fmt(agg.avg_decode_tps)} (median ${fmt(agg.median_decode_tps)}), avg pp=${fmt(agg.avg_prompt_tps, 1)}, prompt_tok=${totalPrompt}, gen_tok=${totalGen}`);
        }
    } catch (e) {
        const rec = Object.assign({}, base, { kind: feed.mode, status: 'FAIL', error: e.message });
        records.push(rec);
        summary = { model: run.model, engine: run.engine, input: run.input, status: 'FAIL', error: e.message };
        console.error('  FAILED: ' + e.message);
    } finally {
        await postShutdown(port);
        const t0 = Date.now();
        while (!exited && Date.now() - t0 < 15000) await sleep(200);
        if (!exited) { try { child.kill(); } catch (_) {} }
        const last = records[records.length - 1];
        if (last) last.server_log_tail = log.slice(-2000);
    }
    return { records, summary };
}

(async () => {
    const p = parseArgv();
    if (!p.models || !p.engines || !p.tasks) {
        console.error('usage: node tools/run_bench.js --models <spec[,...]> --engines <spec[,...]> --tasks <spec[,...]> [--port N] [--health-timeout S] [--out FILE]');
        process.exit(2);
    }
    const port = p.port || Number(process.env.SM_BENCH_PORT || 8994);
    const healthTimeout = p.healthTimeout || 900 * 1000;
    const out = p.out || process.env.SM_BENCH_OUT ||
        path.join(REPO_ROOT, 'benchmark', 'results', 'bench_' + new Date().toISOString().replace(/[:.]/g, '-') + '.jsonl');

    const runs = cartesian(readSpecList(p.models), readSpecList(p.engines), readSpecList(p.tasks));
    console.log('[run_bench] ' + runs.length + ' runs -> ' + out);
    for (const r of runs) console.log('  ' + r.model + '/' + r.engine + '/' + r.input);
    if (process.argv.includes('--dry-run')) { console.log('[run_bench] dry-run, not executing'); return; }

    fs.mkdirSync(path.dirname(out), { recursive: true });
    const summaries = [];
    for (const r of runs) {
        const { records, summary } = await runOne(r, port, healthTimeout);
        for (const rec of records) fs.appendFileSync(out, JSON.stringify(rec) + '\n');
        summaries.push(summary);
    }

    console.log('\n==================== SUMMARY ====================');
    console.log('model            engine                 decode     pp        status');
    for (const s of summaries) {
        console.log(
            String(s.model).padEnd(16) + ' ' + String(s.engine).padEnd(22) + ' ' +
            fmt(s.decode_tps).padStart(8) + ' ' + fmt(s.prompt_tps, 1).padStart(9) + '   ' +
            s.status + (s.error ? ' (' + s.error + ')' : '') +
            (s.detail ? '  [' + s.detail + ']' : '') + (s.turns ? '  [' + s.turns + ' turns]' : '')
        );
    }
    console.log('[run_bench] results: ' + out);
})();
