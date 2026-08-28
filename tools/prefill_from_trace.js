// prefill_from_trace.js - extract a ~N-token real conversation record from the
// captured opencode sequences (temp/sequences, the fake capture server) and send
// it as a prefill to a running llama-server (--expert-backend).
//
// The server must be started with LLM_EXPORT_DIR=<--out> so its export lands in
// the persistent trace/stats directory (expert_history_main.bin + prefill_export_main.bin),
// which is the input for tools/simulate_cache.js policy/capacity tests.
//
// usage:
//   node tools/prefill_from_trace.js [--tokens 10000] [--url http://127.0.0.1:8993]
//                                    [--src temp/sequences] [--out benchmark/trace]
//                                    [--max-tokens 16] [--include-system]
const fs = require('fs');
const path = require('path');
const http = require('http');

function parseArgs() {
    const a = process.argv.slice(2);
    const p = { tokens: 10000, url: 'http://127.0.0.1:8993', src: 'temp/sequences', out: 'benchmark/trace', maxTokens: 16, includeSystem: false };
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '--tokens') p.tokens = parseInt(a[++i]);
        else if (a[i] === '--url') p.url = a[++i];
        else if (a[i] === '--src') p.src = a[++i];
        else if (a[i] === '--out') p.out = a[++i];
        else if (a[i] === '--max-tokens') p.maxTokens = parseInt(a[++i]);
        else if (a[i] === '--include-system') p.includeSystem = true;
    }
    return p;
}

function estTokens(s) {
    s = String(s);
    const cjk = (s.match(/[\u4e00-\u9fff]/g) || []).length;
    return Math.round(cjk / 1.2 + (s.length - cjk) / 4);
}

function textOf(content) {
    if (content == null) return '';
    if (typeof content === 'string') return content;
    if (Array.isArray(content)) return content.map((x) => (x && x.type === 'text' ? x.text : '')).join('\n');
    return JSON.stringify(content);
}

// latest captured POST request (each one carries the full conversation history)
function latestCapture(src) {
    const files = fs.readdirSync(src).filter((f) => f.includes('POST')).sort((a, b) => fs.statSync(path.join(src, a)).mtimeMs - fs.statSync(path.join(src, b)).mtimeMs);
    if (!files.length) throw new Error(`no POST captures in ${src}`);
    return JSON.parse(fs.readFileSync(path.join(src, files[files.length - 1]), 'utf8'));
}

function main() {
    const p = parseArgs();
    const rec = latestCapture(p.src);
    const all = rec.body.messages || [];

    // flatten to {role, content} text-only; drop system (unless asked) and
    // tool_calls-only messages (empty content). tool results fold into user role
    // (llama-server chat API rejects tool messages without a matching tool_call).
    const conv = [];
    for (const m of all) {
        const c = textOf(m.content);
        if (m.role === 'system' && !p.includeSystem) continue;
        if (!c.trim()) continue;
        conv.push({ role: m.role === 'tool' ? 'user' : m.role, content: c });
    }

    // end at the last complete assistant answer (scan from the end)
    let end = conv.length - 1;
    while (end >= 0 && conv[end].role !== 'assistant') end--;
    if (end < 0) end = conv.length - 1;

    // grow [start..end] from the end toward the target token count
    let total = 0, start = end;
    for (let i = end; i >= 0 && total < p.tokens; i--) {
        total += estTokens(conv[i].content);
        start = i;
    }
    const selected = conv.slice(start, end + 1);
    const est = selected.reduce((s, m) => s + estTokens(m.content), 0);
    console.log(`[trace] src=${p.src} messages=${all.length} -> conv=${conv.length}`);
    console.log(`[trace] selected [${start}..${end}] = ${selected.length} msgs, ~${est} tokens (target ${p.tokens})`);

    // snapshot the prompt (persistent, not temp)
    fs.mkdirSync(p.out, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const promptFile = path.join(p.out, `prompt_${est}tok_${ts}.json`);
    fs.writeFileSync(promptFile, JSON.stringify(selected, null, 2));
    console.log(`[trace] prompt snapshot: ${promptFile}`);

    // POST to the running server (prefill + a few generated tokens)
    const body = JSON.stringify({ model: 'deepseek', messages: selected, max_tokens: p.maxTokens, stream: false });
    const u = new URL(p.url);
    const req = http.request({
        hostname: u.hostname, port: u.port, path: '/v1/chat/completions', method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, (res) => {
        let data = '';
        res.on('data', (c) => (data += c));
        res.on('end', () => {
            try {
                const j = JSON.parse(data);
                const usage = j.usage ? `prompt=${j.usage.prompt_tokens} completion=${j.usage.completion_tokens}` : 'no usage';
                const content = j.choices && j.choices[0] && j.choices[0].message ? j.choices[0].message.content : '(empty)';
                console.log(`[trace] POST ${p.url}/v1/chat/completions -> ${res.statusCode} ${usage}`);
                console.log(`[trace] reply: ${String(content).slice(0, 200).replace(/\n/g, ' ')}`);
            } catch (e) {
                console.error(`[trace] response parse failed: ${e.message}`);
                console.error(data.slice(0, 300));
            }
        });
    });
    req.on('error', (e) => {
        console.error(`[trace] POST failed: ${e.message}\n   is llama-server running on ${p.url}?`);
        process.exit(1);
    });
    req.write(body);
    req.end();
    console.log(`[trace] NOTE: server must run with LLM_EXPORT_DIR=${p.out} so expert_history_main.bin lands there.`);
    console.log(`[trace] done.`);
}

main();
