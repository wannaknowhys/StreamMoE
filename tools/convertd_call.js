// convertd_call.js - fire one JSON command at stream_moe_convertd.exe (stdin
// JSON-line, stdout response). Exit 0 iff ok:true. Used by verify_convert_matrix.bat.
// usage: node tools/convertd_call.js '{"cmd":"chunk","in":"...","out":"..."}'
const { spawn } = require('child_process');
const path = require('path');

const EXE = path.join(__dirname, '..', 'temp', 'stream_moe_convertd.exe');
const cmd = process.argv[2];
if (!cmd) { console.error('usage: node tools/convertd_call.js <json>'); process.exit(2); }

const child = spawn(EXE, [], { stdio: ['pipe', 'pipe', 'pipe'] });
let out = '';
child.stdout.on('data', (d) => (out += d));
child.stderr.on('data', (d) => process.stderr.write(d));
child.on('close', () => {
    process.stdout.write(out.trim() + '\n');
    try { process.exit(JSON.parse(out).ok ? 0 : 1); } catch (e) { process.exit(1); }
});
child.on('error', (e) => { console.error('cannot spawn ' + EXE + ': ' + e.message); process.exit(1); });
child.stdin.write(cmd + '\n');
child.stdin.end();
