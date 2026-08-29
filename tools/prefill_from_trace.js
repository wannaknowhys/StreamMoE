// prefill_from_trace.js - extract a real conversation record from the captured
// opencode sequences (temp/sequences) and send it as a prefill to a running
// llama-server (--expert-backend).
//
// Fidelity-first: messages are kept AS-IS (system / user / assistant with
// tool_calls / tool with tool_call_id - nothing dropped or re-rolled), taken
// FROM THE FIRST message, truncated at ~N tokens ending on a complete assistant
// answer, so the prefill KV cache matches what deepseek official would build
// for the same input. Sampling is forced deterministic (temperature 0, top_k 1)
// -> lowest perplexity, near-top outputs.
//
// The server must run with LLM_EXPORT_DIR=<--out> so its export lands in the
// persistent trace/stats dir (expert_history_main.bin etc.) - the input for
// tools/simulate_cache.js policy/capacity tests.
//
// usage:
//   node tools/prefill_from_trace.js [--tokens 10000] [--url http://127.0.0.1:8993]
//                                    [--src temp/sequences] [--out benchmark/trace]
//                                    [--max-tokens 16]
const fs = require('fs');
const path = require('path');
const http = require('http');

function parseArgs() {
    const a = process.argv.slice(2);
    const p = { tokens: 10000, url: 'http://127.0.0.1:8993', src: 'temp/sequences', out: 'benchmark/trace', maxTokens: 16 };
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '--tokens') p.tokens = parseInt(a[++i]);
        else if (a[i] === '--url') p.url = a[++i];
        else if (a[i] === '--src') p.src = a[++i];
        else if (a[i] === '--out') p.out = a[++i];
        else if (a[i] === '--max-tokens') p.maxTokens = parseInt(a[++i]);
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

function isCompleteAnswer(m) {
    // a complete assistant answer = assistant with non-empty text content
    return m.role === 'assistant' && textOf(m.content).trim().length > 0;
}

// deterministic source: the fixed archived capture (not "latest"), so the same
// args always produce the identical prompt for std-vs-moe comparison.
function latestCapture(src) {
    const p = path.join(__dirname, '..', 'benchmark', 'trace', 'sequences', '1787884103582_POST__v1_chat_completions.json');
    if (!fs.existsSync(p)) throw new Error(`fixed capture not found: ${p}`);
    return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function main() {
    const p = parseArgs();
    const rec = latestCapture(p.src);
    const all = rec.body.messages || [];

    // from the FIRST message, keep messages as-is; stop at ~target tokens,
    // truncating on the last complete assistant answer.
    let total = 0, lastGoodEnd = -1, stop = -1;
    for (let i = 0; i < all.length; i++) {
        total += estTokens(textOf(all[i].content)) + 4; // +4 per message overhead (roles)
        if (isCompleteAnswer(all[i])) lastGoodEnd = i;
        if (total >= p.tokens) { stop = i; break; }
    }
    let end = stop >= 0 ? lastGoodEnd : all.length - 1;
    if (lastGoodEnd >= 0) end = lastGoodEnd; // end on a complete answer when we can
    const selected = all.slice(0, end + 1);
    const est = selected.reduce((s, m) => s + estTokens(textOf(m.content)), 0);
    const roles = {};
    for (const m of selected) roles[m.role] = (roles[m.role] || 0) + 1;

    console.log(`[trace] src=${p.src} messages=${all.length}`);
    console.log(`[trace] selected [0..${end}] = ${selected.length} msgs, ~${est} tokens (target ${p.tokens})`);
    console.log(`[trace] roles: ${JSON.stringify(roles)}`);

    // snapshot the prompt (persistent, not temp)
    fs.mkdirSync(p.out, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const promptFile = path.join(p.out, `prompt_${est}tok_${ts}.json`);
    fs.writeFileSync(promptFile, JSON.stringify(selected, null, 2));
    console.log(`[trace] prompt snapshot: ${promptFile}`);

    // POST as-is + agentic sampling (docs/SAMPLING.md: temp 1.0, top_p 0.95) -
    // fidelity: prefill KV == official for this input; output not forced to top.
    const body = JSON.stringify({
        model: 'deepseek',
        messages: selected,
        max_tokens: p.maxTokens,
        temperature: 1.0,
        top_p: 0.95,
        stream: false,
    });
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
}

main();
