// convertd_call.js - fire one command at stream_moe_convertd.exe (stdin JSON
// line, stdout response). Exit 0 iff ok:true. Parameter form keeps JSON out of
// bat quoting hell:
//   node tools/convertd_call.js <cmd> key=value key=value ...
//   e.g. node tools/convertd_call.js chunk "in=R:\x\v2.gguf" "out=a,b,c" "ratio=8:9:9:7:9"
const { spawn } = require('child_process');
const path = require('path');

const EXE = path.join(__dirname, '..', 'temp', 'stream_moe_convertd.exe');
const args = process.argv.slice(2);
const cmd = args[0];
if (!cmd) { console.error('usage: node tools/convertd_call.js <cmd> key=value ...'); process.exit(2); }

const obj = { cmd };
for (const a of args.slice(1)) {
    const i = a.indexOf('=');
    if (i < 0) { console.error('bad arg (need key=value): ' + a); process.exit(2); }
    obj[a.slice(0, i)] = a.slice(i + 1);
}

const child = spawn(EXE, [], { stdio: ['pipe', 'pipe', 'pipe'] });
let out = '';
child.stdout.on('data', (d) => (out += d));
child.stderr.on('data', (d) => process.stderr.write(d));
child.on('close', () => {
    process.stdout.write(out.trim() + '\n');
    try { process.exit(JSON.parse(out).ok ? 0 : 1); } catch (e) { process.exit(1); }
});
child.on('error', (e) => { console.error('cannot spawn ' + EXE + ': ' + e.message); process.exit(1); });
child.stdin.write(JSON.stringify(obj) + '\n');
child.stdin.end();
