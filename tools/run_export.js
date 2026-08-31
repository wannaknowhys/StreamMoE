// run_export.js - cartesian-product export runner.
//
// Three spec categories, flat-merged into one run object. All keys across the
// three must be disjoint; ANY duplicate key on merge aborts (no special-casing).
//   model : { model, modelPath, draft? }
//   engine: { engine, bin|binPath, extra? }
//   task  : { input, feed:{type:prefill|jsonl, ...} }
//
// Usage:
//   node tools/run_export.js --models <spec[,spec]> --engines <spec[,spec]> --tasks <spec[,spec]>
//
// Env:
//   SM_OUT_ROOT  export root (default <repo>/temp/exports)
//   SM_PORT      server port (default 8993)
//   ${VAR} in spec files expands from env (machine paths stay out of specs)
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = Number(process.env.SM_PORT || 8993);
const OUT_ROOT = process.env.SM_OUT_ROOT || path.join(__dirname, '..', 'temp', 'exports');

function parseArgv() {
    const a = process.argv.slice(2);
    const p = {};
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '--models') p.models = a[++i];
        else if (a[i] === '--engines') p.engines = a[++i];
        else if (a[i] === '--tasks') p.tasks = a[++i];
    }
    return p;
}

function readSpecList(csv) {
    return csv.split(',').map((f) => JSON.parse(fs.readFileSync(f.trim(), 'utf8')));
}

// Resolve ${X}: run field wins, then env. Called AFTER merge so ${pool} (a
// model spec field) resolves against the merged run.
function expandRun(run) {
    const expand = (s) => s.replace(/\$\{([A-Za-z0-9_]+)\}/g, (m, k) => {
        if (run[k] !== undefined) return String(run[k]);
        if (process.env[k] !== undefined) return process.env[k];
        throw new Error('[run_export] unresolvable ${' + k + '}');
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
                if (k in run) throw new Error('[run_export] duplicate key across specs: ' + k);
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
    if (!run.bin) throw new Error('[run_export] run.bin or run.binPath required');
    return path.join(__dirname, '..', 'build', run.bin, 'llama-build', 'bin', 'llama-server.exe');
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function waitHealth(timeoutMs) {
    const t0 = Date.now();
    return new Promise((resolve, reject) => {
        const poll = () => {
            const req = http.get(`http://127.0.0.1:${PORT}/health`, (res) => { res.resume(); res.statusCode === 200 ? resolve() : retry(); });
            req.on('error', retry);
        };
        const retry = () => (Date.now() - t0 > timeoutMs ? reject(new Error('health timeout')) : setTimeout(poll, 1000));
        poll();
    });
}

function postShutdown() {
    return new Promise((resolve, reject) => {
        const req = http.request(`http://127.0.0.1:${PORT}/shutdown`, { method: 'POST' }, (res) => { res.resume(); res.on('end', () => resolve()); });
        req.on('error', reject);
        req.end();
    });
}

function postChat(messages) {
    return new Promise((resolve, reject) => {
        const body = JSON.stringify({ model: 'm', messages, max_tokens: 1024, stream: false });
        const req = http.request({ hostname: '127.0.0.1', port: PORT, path: '/v1/chat/completions', method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (res) => {
            let d = '';
            res.on('data', (c) => (d += c));
            res.on('end', () => {
                try {
                    const j = JSON.parse(d);
                    if (j.error) resolve({ error: String(j.error.message || j.error) });
                    else {
                        const msg = j.choices && j.choices[0] && j.choices[0].message;
                        // keep the full message object (reasoning_content + content +
                        // tool_calls etc.) so chat.json can replay exactly what the
                        // model emitted, even when truncated / ctx-exhausted.
                        resolve({
                            msg: msg || null,
                            content: msg ? (msg.content || '') : '',
                            prompt_tokens: j.usage ? j.usage.prompt_tokens : 0,
                        });
                    }
                } catch (e) { reject(new Error('bad chat JSON: ' + d.slice(0, 200))); }
            });
        });
        req.on('error', reject);
        req.end(body);
    });
}

async function feedTask(run, dir) {
    const f = run.feed;
    if (!f || !f.type) throw new Error('[run_export] run.feed.type required (prefill|jsonl)');
    if (f.type === 'prefill') {
        // prompt snapshot (the exact sent payload) lands in the run dir next to
        // the exported bins (prefill_from_trace writes it to --out).
        const tokens = String(f.tokens || 10000);
        const c = spawn(process.execPath, [path.join(__dirname, 'prefill_from_trace.js'), '--tokens', tokens, '--out', dir], { stdio: 'inherit' });
        await new Promise((res, rej) => { c.on('exit', (code) => (code === 0 ? res() : rej(new Error('prefill_from_trace exit ' + code)))); });
    } else if (f.type === 'jsonl') {
        const lines = fs.readFileSync(f.path, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
        const maxTurns = f.maxTurns || lines.length;
        const messages = [];   // request body (final content only)
        const chatLog = [];    // full message objects incl. reasoning_content
        for (let i = 0; i < maxTurns; i++) {
            const { system, prompt } = lines[i];
            if (system && String(system).trim()) { messages.push({ role: 'system', content: system }); chatLog.push({ role: 'system', content: system }); }
            messages.push({ role: 'user', content: prompt });
            chatLog.push({ role: 'user', content: prompt });
            const res = await postChat(messages);
            if (res.error) { console.log('  turn ' + (i + 1) + '/' + maxTurns + ': ERROR ' + res.error); break; }
            messages.push({ role: 'assistant', content: res.content });
            chatLog.push(res.msg || { role: 'assistant', content: res.content });
            console.log('  turn ' + (i + 1) + '/' + maxTurns + ' done (prompt_tokens=' + res.prompt_tokens + ')');
        }
        // full conversation incl. complete model messages (thinking + content),
        // next to the exported bins
        fs.writeFileSync(path.join(dir, 'chat.json'), JSON.stringify(chatLog, null, 2));
        console.log('  chat saved: ' + path.join(dir, 'chat.json') + ' (' + chatLog.length + ' msgs)');
    } else throw new Error('[run_export] unknown feed type ' + f.type);
}

async function runOne(run) {
    const dir = path.join(OUT_ROOT, run.model, run.engine, run.input);
    fs.mkdirSync(dir, { recursive: true });
    const bin = binPath(run);
    const args = ['-m', run.modelPath, '--fit', 'off', '--no-warmup', '-c', '15000', '-t', '16',
        '--top-p', '0.95', '--host', '127.0.0.1', '--port', String(PORT), '--no-webui'];
    if (run.draft) args.push('--model-draft', run.draft, '--spec-draft-n-max', '5', '--spec-draft-n-min', '1', '--spec-draft-p-min', '0.6');
    if (run.extra) args.push(...run.extra);
    args.push('--export-dir', dir); // StreamMoE: export via arg (replaces LLM_EXPORT_DIR env)
    console.log(`\n=== ${path.basename(bin)} ${run.model}/${run.engine}/${run.input} -> ${dir} ===`);
    const child = spawn(bin, args, { stdio: ['ignore', 'inherit', 'inherit'] });
    try {
        await waitHealth(600000);
        await feedTask(run, dir);
        await postShutdown();
        await new Promise((res, rej) => { child.on('exit', res); child.on('error', rej); });
        console.log('  ' + dir + ' done');
    } catch (e) {
        console.error('  FAILED: ' + e.message);
        try { child.kill(); } catch (_) {}
        process.exit(1);
    }
}

(async () => {
    const p = parseArgv();
    if (!p.models || !p.engines || !p.tasks) {
        console.error('usage: node tools/run_export.js --models <spec[,...]> --engines <spec[,...]> --tasks <spec[,...]>');
        process.exit(2);
    }
    const dry = process.argv.includes('--dry-run');
    const runs = cartesian(readSpecList(p.models), readSpecList(p.engines), readSpecList(p.tasks));
    console.log('[run_export] ' + runs.length + ' runs:');
    for (const r of runs) console.log('  ' + r.model + '/' + r.engine + '/' + r.input + '  ->  ' + path.join(OUT_ROOT, r.model, r.engine, r.input));
    if (dry) { console.log('[run_export] dry-run, not executing'); return; }
    for (const r of runs) await runOne(r);
    console.log('\n[run_export] ALL DONE');
})();
