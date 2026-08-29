// run_export_orchestration.js - std-vs-moe export orchestration (merged from
// run_export_pair.js + run_longhorizon_export.js).
//
// Modes (first CLI arg):
//   pair        moe (build/rbexport, --expert-backend) then upstream (build/upstream)
//               short-conversation prefill via tools/prefill_from_trace.js
//   longhorizon waits for port free, then 4 rounds of long-horizon jsonl
//               round-trips: moe{cn,en} + upstream{cn,en}
//   all         pair, then longhorizon (same process, port reused sequentially)
//
// Environment (machine-specific, NOT hardcoded):
//   SM_MODEL    main GGUF path (required)
//   SM_DRAFT    draft GGUF path  (required)
//   SM_OUT_ROOT export root; defaults to temp/exports under this repo
//   SM_PORT     server port; default 8993
//
// Example:
//   $env:SM_MODEL='N:\AI_LLM\...\main.gguf'; $env:SM_DRAFT='N:\...\draft.gguf'
//   node tools/run_export_orchestration.js all
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = Number(process.env.SM_PORT || 8993);
const MODEL = process.env.SM_MODEL;
const DRAFT = process.env.SM_DRAFT;
const OUT_ROOT = process.env.SM_OUT_ROOT || path.join(__dirname, '..', 'temp', 'exports');
const MAX_TOKENS = 256;
const CTX = 15000;

const BIN_MOE = path.join(__dirname, '..', 'build', 'rbexport', 'llama-build', 'bin', 'llama-server.exe');
const BIN_UP  = path.join(__dirname, '..', 'build', 'upstream', 'llama-build', 'bin', 'llama-server.exe');
const JSONL_CN = path.join(__dirname, '..', 'benchmark', 'prompts', 'long_horizon_prompts_zh.jsonl');
const JSONL_EN = path.join(__dirname, '..', 'benchmark', 'prompts', 'long_horizon_prompts.jsonl');

if (!MODEL || !DRAFT) {
    console.error('[orchestration] SM_MODEL and SM_DRAFT env vars are required');
    process.exit(2);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function waitHealth(url, timeoutMs) {
    const start = Date.now();
    return new Promise((resolve, reject) => {
        const poll = () => {
            const req = http.get(url, (res) => { res.resume(); res.statusCode === 200 ? resolve(true) : retry(); });
            req.on('error', retry);
        };
        const retry = () => (Date.now() - start > timeoutMs ? reject(new Error('health timeout: ' + url)) : setTimeout(poll, 1000));
        poll();
    });
}

function post(url) {
    return new Promise((resolve, reject) => {
        const req = http.request(url, { method: 'POST' }, (res) => { res.resume(); res.on('end', () => resolve(res.statusCode)); });
        req.on('error', reject);
        req.end();
    });
}

function postChat(messages) {
    return new Promise((resolve, reject) => {
        const body = JSON.stringify({ model: 'deepseek', messages, max_tokens: MAX_TOKENS, stream: false });
        const req = http.request({ hostname: '127.0.0.1', port: PORT, path: '/v1/chat/completions', method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (res) => {
            let d = '';
            res.on('data', (c) => (d += c));
            res.on('end', () => {
                try {
                    const j = JSON.parse(d);
                    if (j.error) resolve({ error: String(j.error.message || j.error) });
                    else resolve({ content: j.choices && j.choices[0] && j.choices[0].message ? j.choices[0].message.content || '' : '', prompt_tokens: j.usage ? j.usage.prompt_tokens : 0 });
                } catch (e) { reject(new Error('bad chat JSON: ' + d.slice(0, 200))); }
            });
        });
        req.on('error', reject);
        req.end(body);
    });
}

function serverArgs(extra) {
    const base = ['-m', MODEL, '--model-draft', DRAFT,
        '--spec-draft-n-max', '5', '--spec-draft-n-min', '1', '--spec-draft-p-min', '0.6',
        '--fit', 'off', '--no-warmup', '-c', String(CTX), '-t', '16',
        '--temp', '1.0', '--top-p', '0.95', '--metrics',
        '--host', '127.0.0.1', '--port', String(PORT), '--no-webui'];
    return base.concat(extra);
}

// spawn a server, wait healthy, run a job, graceful /shutdown, wait exit.
async function runServer(bin, dir, extra, job) {
    fs.mkdirSync(dir, { recursive: true });
    console.log(`\n[export] === ${path.basename(bin)} -> ${dir} ===`);
    const child = spawn(bin, serverArgs(extra), {
        stdio: ['ignore', 'inherit', 'inherit'],
        env: { ...process.env, LLM_EXPORT_DIR: dir },
    });
    try {
        await waitHealth(`http://127.0.0.1:${PORT}/health`, 600000);
        await job();
        console.log(`[export] job done, POST /shutdown (graceful export flush)`);
        await post(`http://127.0.0.1:${PORT}/shutdown`);
        await new Promise((resolve, reject) => { child.on('exit', resolve); child.on('error', reject); });
        console.log(`[export] ${dir} done, exports written`);
    } catch (e) {
        console.error(`[export] FAILED for ${dir}: ${e.message}`);
        try { child.kill(); } catch (_) {}
        process.exit(1);
    }
}

// pair mode: deterministic prefill (same capture) through moe then upstream.
async function runPrefill() {
    const c = spawn(process.execPath, [path.join(__dirname, 'prefill_from_trace.js'), '--tokens', '10000'], { stdio: 'inherit' });
    await new Promise((resolve, reject) => { c.on('exit', (code) => (code === 0 ? resolve() : reject(new Error('prefill_from_trace exit ' + code)))); });
}

async function modePair() {
    await runServer(BIN_MOE, path.join(OUT_ROOT, 'moe'), ['--expert-backend', '--moe-ram-pool', '71680'], runPrefill);
    await runServer(BIN_UP, path.join(OUT_ROOT, 'upstream'), [], runPrefill);
    console.log(`\n[export] PAIR DONE (${path.join(OUT_ROOT, 'moe')} + ${path.join(OUT_ROOT, 'upstream')})`);
}

// longhorizon mode: round-trip feed accumulating context, report ctx overflow.
async function runRoundTrip(bin, dir, jsonlPath, extra) {
    await runServer(bin, dir, extra, async () => {
        console.log('  healthy, round-trip feeding...');
        const lines = fs.readFileSync(jsonlPath, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
        let messages = [];
        let overflowAt = null;
        for (let i = 0; i < lines.length; i++) {
            const { system, prompt } = lines[i];
            messages.push({ role: 'system', content: system });
            messages.push({ role: 'user', content: prompt });
            const res = await postChat(messages);
            if (res.error) {
                console.log(`  turn ${i + 1}/${lines.length}: ERROR ${res.error}`);
                if (/context|ctx|overflow|window/i.test(res.error) && overflowAt === null) overflowAt = i + 1;
                break;
            }
            messages.push({ role: 'assistant', content: res.content });
            console.log(`  turn ${i + 1}/${lines.length} done (prompt_tokens=${res.prompt_tokens})`);
        }
        if (overflowAt !== null) console.log(`  !! ctx overflow at turn ${overflowAt} - -c ${CTX} too small`);
    });
}

async function waitPortFree(port, timeoutMs) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
        const up = await new Promise((resolve) => {
            const req = http.get(`http://127.0.0.1:${port}/health`, (res) => { res.resume(); resolve(true); });
            req.on('error', () => resolve(false));
        });
        if (!up) return;
        console.log(`  [wait] port ${port} busy, retry in 60s...`);
        await sleep(60000);
    }
    throw new Error('port ' + port + ' not free within ' + timeoutMs + 'ms');
}

async function modeLonghorizon() {
    console.log(`[longhorizon] waiting for port ${PORT} to free...`);
    await waitPortFree(PORT, 7200000); // up to 2h
    const runs = [
        [BIN_MOE, 'moe_cn', JSONL_CN, ['--expert-backend', '--moe-ram-pool', '71680']],
        [BIN_MOE, 'moe_en', JSONL_EN, ['--expert-backend', '--moe-ram-pool', '71680']],
        [BIN_UP,  'upstream_cn', JSONL_CN, []],
        [BIN_UP,  'upstream_en', JSONL_EN, []],
    ];
    for (const [bin, sub, jsonl, extra] of runs) {
        await runRoundTrip(bin, path.join(OUT_ROOT, sub), jsonl, extra);
    }
    console.log(`\n[longhorizon] ALL DONE (16 bins under ${OUT_ROOT})`);
}

async function main() {
    const mode = process.argv[2] || 'pair';
    if (mode === 'pair') await modePair();
    else if (mode === 'longhorizon') await modeLonghorizon();
    else if (mode === 'all') { await modePair(); await modeLonghorizon(); }
    else { console.error(`unknown mode '${mode}' (use pair | longhorizon | all)`); process.exit(2); }
}
main();
