// run_export.js - 4-dimensional cartesian-product export runner.
//
// Four spec categories, flat-merged into one run object. All keys across the
// four must be disjoint; ANY duplicate key on merge aborts (no special-casing).
//   flavor: { flavor, bin|binPath, hasExport, hasRouteB, desc? }
//   model : { model, modelPath, draft?, pool?, modelCtx? }
//   engine: { engine, extra? }
//   task  : { input, feed:{type:prefill|jsonl, ...}, taskCtx? }
//
// Usage:
//   node tools/run_export.js --flavors <spec[,spec]> --models <spec[,spec]> --engines <spec[,spec]> --tasks <spec[,spec]>
//
// Env:
//   SM_OUT_ROOT  export root (default <repo>/temp/exports)
//   SM_PORT      server port (default 8993)
//   ${VAR} in spec files expands from env (machine paths stay out of specs)
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { checkBinaryFreshness } = require('./freshness_check');

const REPO_ROOT = path.resolve(__dirname, '..');
const PORT = Number(process.env.SM_PORT || 8993);
const OUT_ROOT = process.env.SM_OUT_ROOT || path.join(REPO_ROOT, 'temp', 'exports');

function loadEnvIfPresent() {
    const envBat = path.join(REPO_ROOT, 'temp', 'sm_env.bat');
    if (fs.existsSync(envBat)) {
        const lines = fs.readFileSync(envBat, 'utf8').split('\n');
        for (const line of lines) {
            const m = line.trim().match(/^set\s+([A-Za-z0-9_]+)=(.*)$/i);
            if (m && m[1] && m[2]) {
                if (process.env[m[1].trim()] === undefined) {
                    process.env[m[1].trim()] = m[2].trim();
                }
            }
        }
    }
}
loadEnvIfPresent();

const SPEC_DIRS = {
    flavors: path.join(__dirname, 'run_specs', 'flavors'),
    models: path.join(__dirname, 'run_specs', 'models'),
    engines: path.join(__dirname, 'run_specs', 'engines'),
    tasks: path.join(__dirname, 'run_specs', 'tasks'),
};

function parseArgv() {
    const a = process.argv.slice(2);
    const p = {};
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '--flavors') p.flavors = a[++i];
        else if (a[i] === '--models') p.models = a[++i];
        else if (a[i] === '--engines') p.engines = a[++i];
        else if (a[i] === '--tasks') p.tasks = a[++i];
        else if (a[i] === '--bin-override') p.binOverride = a[++i];
        else if (a[i] === '--temp-override') p.tempOverride = a[++i];
        else if (a[i] === '--out-root') p.outRoot = a[++i];
        else if (a[i] === '--force-export') p.forceExport = true;
    }
    return p;
}

function resolveSpecPath(input, category) {
    const trimmed = input.trim();
    if (fs.existsSync(trimmed)) return path.resolve(trimmed);
    const dir = SPEC_DIRS[category];
    if (dir) {
        const withExt = path.join(dir, trimmed.endsWith('.json') ? trimmed : trimmed + '.json');
        if (fs.existsSync(withExt)) return withExt;
    }
    throw new Error(`[run_export] Cannot resolve ${category} spec: "${trimmed}"`);
}

function readSpecList(csv, category) {
    if (!csv) return [];
    return csv.split(',').map((f) => {
        const resolved = resolveSpecPath(f, category);
        return JSON.parse(fs.readFileSync(resolved, 'utf8').replace(/^\uFEFF/, ''));
    });
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

function cartesian(flavors, models, engines, tasks) {
    const out = [];
    for (const fl of flavors) {
        for (const m of models) {
            for (const e of engines) {
                for (const t of tasks) {
                    const run = {};
                    const merge = (src) => {
                        for (const [k, v] of Object.entries(src)) {
                            if (k in run) throw new Error('[run_export] duplicate key across specs: ' + k);
                            run[k] = v;
                        }
                    };
                    merge(fl);
                    merge(m);
                    merge(e);
                    merge(t);
                    out.push(expandRun(run));
                }
            }
        }
    }
    return out;
}

function binPath(run, binOverride) {
    if (binOverride) return path.resolve(binOverride);
    if (run.binPath) return path.resolve(run.binPath);
    const tag = run.bin || run.flavor;
    if (!tag) throw new Error('[run_export] run.bin or run.flavor required');
    return path.join(REPO_ROOT, 'build', tag, 'llama-build', 'bin', 'llama-server.exe');
}

function computeEffectiveContext(run) {
    const m = Number(run.modelCtx);
    const t = Number(run.taskCtx || run.ctx);
    const reqTokens = (run.feed && Number.isFinite(Number(run.feed.tokens))) ? Number(run.feed.tokens) : 0;

    if (Number.isFinite(m) && Number.isFinite(t)) {
        if (reqTokens > 0 && m < reqTokens) {
            console.warn(`[run_export] WARNING: task requires ${reqTokens} tokens (taskCtx=${t}), exceeding modelCtx=${m}. Preserving taskCtx ${t} to prevent truncation.`);
            return t;
        }
        if (t > m) {
            console.warn(`[run_export] WARNING: taskCtx (${t}) exceeds modelCtx (${m}). Clamping to min(${m}, ${t}) = ${Math.min(m, t)}.`);
        }
        return Math.min(m, t);
    }
    if (Number.isFinite(m)) return m;
    if (Number.isFinite(t)) return t;
    return 4096;
}

function buildServerArgs(run, bin, dir, opts = {}) {
    const effectiveCtx = computeEffectiveContext(run);
    const threads = Number(run.threads || 16);

    // Parse extra options into a structured map for deduplication
    const extra = Array.isArray(run.extra) ? [...run.extra] : [];
    const extraMap = new Map();
    const positionalExtra = [];

    for (let i = 0; i < extra.length; i++) {
        const item = extra[i];
        if (typeof item === 'string' && item.startsWith('-')) {
            if (i + 1 < extra.length && !extra[i + 1].startsWith('-')) {
                extraMap.set(item, extra[++i]);
            } else {
                extraMap.set(item, true);
            }
        } else {
            positionalExtra.push(item);
        }
    }

    // Determine temperature override
    let tempVal = null;
    if (opts.tempOverride !== undefined && opts.tempOverride !== null) {
        tempVal = String(opts.tempOverride);
    } else if (extraMap.has('--temp')) {
        tempVal = String(extraMap.get('--temp'));
    }

    // Determine context size
    let ctxVal = String(effectiveCtx);
    if (extraMap.has('-c')) {
        const extraC = Number(extraMap.get('-c'));
        if (Number.isFinite(extraC)) ctxVal = String(Math.min(effectiveCtx, extraC));
    }

    // Determine --fit
    const fitVal = extraMap.has('--fit') ? String(extraMap.get('--fit')) : 'off';

    // Base CLI arguments
    const args = [
        '-m', run.modelPath,
        '-c', ctxVal,
        '-t', String(threads),
        '--fit', fitVal,
        '--no-warmup',
        '--host', '127.0.0.1',
        '--port', String(PORT),
        '--no-webui'
    ];

    if (tempVal !== null) {
        args.push('--temp', tempVal);
    }

    if (run.draft) {
        args.push(
            '--model-draft', run.draft,
            '--spec-draft-n-max', '5',
            '--spec-draft-n-min', '1',
            '--spec-draft-p-min', '0.6'
        );
    }

    // Append remaining extra options (excluding handled keys)
    const handledKeys = new Set(['--temp', '-c', '--ctx-size', '--fit', '-t', '--threads', '-m', '--model', '--host', '--port', '--webui', '--no-webui']);
    for (const [k, v] of extraMap.entries()) {
        if (handledKeys.has(k)) continue;
        args.push(k);
        if (v !== true) args.push(String(v));
    }
    args.push(...positionalExtra);

    // Export directory: ONLY passed when the flavor supports export AND
    // the task is a prefill export task or explicit export is requested.
    const isPrefillTask = run.feed && run.feed.type === 'prefill';
    const shouldExport = Boolean(run.hasExport && (isPrefillTask || opts.forceExport));
    if (shouldExport) {
        args.push('--export-dir', dir);
    }

    return args;
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

function postChat(messages, tempOverride, maxTokens) {
    return new Promise((resolve, reject) => {
        const bodyObj = { model: 'm', messages, max_tokens: maxTokens || 1024, stream: false };
        if (tempOverride !== undefined && tempOverride !== null) {
            bodyObj.temperature = Number(tempOverride);
        }
        const body = JSON.stringify(bodyObj);
        const req = http.request({ hostname: '127.0.0.1', port: PORT, path: '/v1/chat/completions', method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (res) => {
            let d = '';
            res.on('data', (c) => (d += c));
            res.on('end', () => {
                try {
                    const j = JSON.parse(d);
                    if (j.error) resolve({ error: String(j.error.message || j.error) });
                    else {
                        const msg = j.choices && j.choices[0] && j.choices[0].message;
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

async function feedTask(run, dir, tempOverride) {
    const f = run.feed;
    if (!f || !f.type) throw new Error('[run_export] run.feed.type required (prefill|jsonl)');
    if (f.type === 'prefill') {
        const tokens = String(f.tokens || 10000);
        const c = spawn(process.execPath, [path.join(__dirname, 'prefill_from_trace.js'), '--tokens', tokens, '--out', dir], { stdio: 'inherit' });
        await new Promise((res, rej) => { c.on('exit', (code) => (code === 0 ? res() : rej(new Error('prefill_from_trace exit ' + code)))); });
    } else if (f.type === 'jsonl') {
        const lines = fs.readFileSync(f.path, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
        const maxTurns = f.maxTurns || lines.length;
        const messages = [];
        const chatLog = [];
        for (let i = 0; i < maxTurns; i++) {
            const { system, prompt } = lines[i];
            if (system && String(system).trim()) { messages.push({ role: 'system', content: system }); chatLog.push({ role: 'system', content: system }); }
            messages.push({ role: 'user', content: prompt });
            chatLog.push({ role: 'user', content: prompt });
            const res = await postChat(messages, tempOverride, f.maxTokens);
            if (res.error) { console.log('  turn ' + (i + 1) + '/' + maxTurns + ': ERROR ' + res.error); break; }
            messages.push({ role: 'assistant', content: res.content });
            chatLog.push(res.msg || { role: 'assistant', content: res.content });
            console.log('  turn ' + (i + 1) + '/' + maxTurns + ' done (prompt_tokens=' + res.prompt_tokens + ')');
        }
        fs.writeFileSync(path.join(dir, 'chat.json'), JSON.stringify(chatLog, null, 2));
        console.log('  chat saved: ' + path.join(dir, 'chat.json') + ' (' + chatLog.length + ' msgs)');
    } else throw new Error('[run_export] unknown feed type ' + f.type);
}

async function runOne(run, opts = {}) {
    const outRoot = opts.outRoot || OUT_ROOT;
    const dir = path.join(outRoot, run.flavor || 'default', run.model, run.engine, run.input);
    fs.mkdirSync(dir, { recursive: true });
    const bin = binPath(run, opts.binOverride);
    checkBinaryFreshness(bin);

    const args = buildServerArgs(run, bin, dir, opts);
    console.log(`\n=== ${path.basename(bin)} [${run.flavor}] ${run.model}/${run.engine}/${run.input} -> ${dir} ===`);
    console.log(`  Args: ${args.join(' ')}`);

    const child = spawn(bin, args, { stdio: ['ignore', 'inherit', 'inherit'] });
    try {
        await waitHealth(600000);
        await feedTask(run, dir, opts.tempOverride);
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
        console.error('usage: node tools/run_export.js [--flavors <spec[,...]>] --models <spec[,...]> --engines <spec[,...]> --tasks <spec[,...]> [--bin-override <path>] [--temp-override <val>] [--out-root <dir>]');
        process.exit(2);
    }
    const dry = process.argv.includes('--dry-run');

    // Default flavor is StreamMoE if omitted
    const flavorsCsv = p.flavors || 'StreamMoE';
    const flavors = readSpecList(flavorsCsv, 'flavors');
    const models = readSpecList(p.models, 'models');
    const engines = readSpecList(p.engines, 'engines');
    const tasks = readSpecList(p.tasks, 'tasks');

    const runs = cartesian(flavors, models, engines, tasks);
    const effectiveOut = p.outRoot || OUT_ROOT;
    console.log('[run_export] ' + runs.length + ' runs:');
    for (const r of runs) {
        console.log(`  [${r.flavor}] ${r.model}/${r.engine}/${r.input}  ->  ${path.join(effectiveOut, r.flavor, r.model, r.engine, r.input)}`);
    }
    if (dry) { console.log('[run_export] dry-run, not executing'); return; }
    for (const r of runs) await runOne(r, p);
    console.log('\n[run_export] ALL DONE');
})();
