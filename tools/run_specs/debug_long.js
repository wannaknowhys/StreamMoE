const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const bin = 'build/upstream_dump/llama-build/bin/llama-server.exe';
const line = fs.readFileSync('benchmark/prompts/long_horizon_prompts_zh.jsonl', 'utf8').split('\n')[0];
const prompt = JSON.parse(line).prompt;
console.log('prompt chars=' + prompt.length + ' (~131 tokens, build.bat upstream_dump VULKAN=ON)');
const args = ['-m', 'N:/AI_LLM/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf', '--fit', 'off', '--no-warmup', '-c', '4096', '-t', '8',
  '--n-gpu-layers', '0', '--host', '127.0.0.1', '--port', '8999', '--no-webui', '--temp', '1', '--top-p', '0.95',
  '--export-dir', 'O:/1/dbg_export'];
const child = spawn(bin, args, { stdio: ['ignore', 'pipe', 'pipe'] });
let serverLog = '';
child.stdout.on('data', (d) => (serverLog += d));
child.stderr.on('data', (d) => (serverLog += d));
child.on('exit', (c, s) => { console.log('SERVER EXIT code=' + c); fs.writeFileSync('temp/sched_full.log', serverLog); process.exit(c); });
function healthOk() {
  return new Promise((res) => {
    const req = http.get('http://127.0.0.1:8999/health', (r) => { r.resume(); r.on('end', () => res(r.statusCode === 200)); });
    req.on('error', () => res(false));
  });
}
async function main() {
  const t0 = Date.now();
  while (Date.now() - t0 < 600000) {
    if (await healthOk()) break;
    await new Promise((r) => setTimeout(r, 1000));
  }
  if (!(await healthOk())) { console.log('HEALTH TIMEOUT'); child.kill(); return; }
  console.log('HEALTH OK - sending long prompt...');
  const body = JSON.stringify({ model: 'm', messages: [{ role: 'user', content: prompt }], max_tokens: 16, stream: false });
  const req = http.request({ hostname: '127.0.0.1', port: 8999, path: '/v1/chat/completions', method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) } }, (res) => {
    let d = ''; res.on('data', (c) => (d += c)); res.on('end', () => {
      console.log('REQUEST OK status=' + res.statusCode + ' len=' + d.length);
      const s = http.request({ hostname: '127.0.0.1', port: 8999, path: '/shutdown', method: 'POST' }, (sr) => { sr.resume(); sr.on('end', () => console.log('shutdown sent')); });
      s.on('error', (e) => console.log('shutdown err ' + e.message));
      s.end();
    });
  });
  req.on('error', (e) => { console.log('REQUEST ERROR: ' + e.message); });
  req.end(body);
  setTimeout(() => {
    console.log('--- serverLog tail ---');
    console.log(serverLog.slice(-2000));
    child.kill();
    process.exit(0);
  }, 600000);
}
main();
