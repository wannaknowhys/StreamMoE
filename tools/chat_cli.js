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

function postStream(payload, onDelta) {
    return new Promise((resolve, reject) => {
        const body = JSON.stringify(payload);
        const req = http.request({
            hostname: host, port, path: '/v1/chat/completions', method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
        }, (res) => {
            if (res.statusCode !== 200) {
                let err = '';
                res.on('data', (c) => (err += c));
                res.on('end', () => reject(new Error(`HTTP ${res.statusCode}: ${err.slice(0, 200)}`)));
                return;
            }
            let buf = '';
            res.on('data', (chunk) => {
                buf += chunk;
                let idx;
                while ((idx = buf.indexOf('\n')) >= 0) {
                    const line = buf.slice(0, idx);
                    buf = buf.slice(idx + 1);
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data.includes('[DONE]')) continue;
                        try {
                            const j = JSON.parse(data);
                            const d = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
                            if (d) onDelta(d);
                        } catch (e) {}
                    }
                }
            });
            res.on('end', () => { if (buf.trim()) { try { const j = JSON.parse(buf.replace(/^data: /, '')); const d = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content; if (d) onDelta(d); } catch (e) {} } resolve(); });
        });
        req.on('error', reject);
        req.write(body);
        req.end();
    });
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
        let reply = '';
        await postStream({ model: 'deepseek4', messages, stream: true }, (piece) => { reply += piece; process.stdout.write(piece); });
        messages.push({ role: 'assistant', content: reply });
        console.log(`\n[${Date.now() - t0} ms]`);
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
