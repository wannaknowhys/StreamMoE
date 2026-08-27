// chat_cli.js - interactive multi-turn chat client for the StreamMoE server.
// Usage: node tools/chat_cli.js [--host 127.0.0.1] [--port 8992]
// Commands:
//   /quit | /exit | /bye   quit (after any in-flight request completes)
//   /reset                 clear conversation history
//   /stats                 show server stats
// Any other line is a user message (streamed reply). Input lines are processed
// strictly in order (a serial queue), so a /quit cannot cut off a live request.
const http = require('http');
const readline = require('readline');

const args = process.argv.slice(2);
let host = '127.0.0.1', port = 8992;
for (let i = 0; i < args.length; ++i) {
    if (args[i] === '--host' && i + 1 < args.length) host = args[++i];
    else if (args[i] === '--port' && i + 1 < args.length) port = parseInt(args[++i], 10);
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
rl.setPrompt('> ');
let messages = [];
let wantExit = false;
let rlClosed = false;
const queue = [];
let pumping = false;

function post(payload) {
    return new Promise((resolve, reject) => {
        const body = JSON.stringify(payload);
        const req = http.request({
            hostname: host, port, path: '/v1/chat/completions', method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
        }, (res) => {
            let data = '';
            res.on('data', (c) => (data += c));
            res.on('end', () => resolve(data));
        });
        req.on('error', reject);
        req.write(body);
        req.end();
    });
}
function extractStream(sse) {
    let out = '';
    for (const line of sse.split('\n')) {
        if (line.startsWith('data: ') && !line.includes('[DONE]')) {
            try {
                const c = JSON.parse(line.substring(6));
                if (c.choices && c.choices[0] && c.choices[0].delta && c.choices[0].delta.content) out += c.choices[0].delta.content;
            } catch (e) {}
        }
    }
    return out.trim();
}
function stats() {
    return new Promise((resolve, reject) => {
        http.get(`http://${host}:${port}/stats`, (res) => {
            let d = '';
            res.on('data', (c) => (d += c));
            res.on('end', () => { try { resolve(JSON.parse(d)); } catch (e) { resolve({}); } });
        }).on('error', reject);
    });
}

async function handle(t) {
    if (/^\/(quit|exit|bye)$/i.test(t)) { wantExit = true; console.log('bye'); rl.close(); return; }
    if (t === '/reset') { messages = []; console.log('[reset] history cleared'); return; }
    if (t === '/stats') {
        const s = await stats();
        console.log(`[stats] model=${s.model} pool=${s.pool_total_mb}MB slots=${s.slots_total} state=${s.runtime_state}`);
        return;
    }
    if (!t) return;
    messages.push({ role: 'user', content: t });
    const t0 = Date.now();
    process.stdout.write('[thinking...] ');
    try {
        const resp = await post({ model: 'deepseek4', messages, stream: true });
        const reply = extractStream(resp);
        messages.push({ role: 'assistant', content: reply });
        console.log(`\n[assistant] ${reply}\n[${Date.now() - t0} ms]`);
    } catch (e) {
        console.error(`\n[error] ${e.message}`);
    }
}

async function pump() {
    if (pumping) return;
    pumping = true;
    while (queue.length && !wantExit) {
        const t = queue.shift();
        await handle(t);
    }
    pumping = false;
    if (wantExit || rlClosed) process.exit(0);
}

rl.on('line', (line) => { queue.push(line.trim()); pump(); });
rl.on('close', () => { rlClosed = true; if (!pumping) process.exit(0); });

console.log(`StreamMoE chat @ http://${host}:${port}  (commands: /quit /reset /stats)`);
rl.prompt();
