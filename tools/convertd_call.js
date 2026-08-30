// convertd_call.js - fire one raw JSON command at stream_moe_convertd.exe over
// TCP (spawns a fresh convertd on an ephemeral port). Exit 0 iff ok:true.
// Two forms:
//   node tools/convertd_call.js '{"cmd":"open","in":["a.gguf"]}'
//   node tools/convertd_call.js open path=...  (key=value form, JSON built here)
const { spawn } = require('child_process');
const net = require('net');
const path = require('path');

const EXE = path.join(__dirname, '..', 'temp', 'stream_moe_convertd.exe');

function main() {
    const args = process.argv.slice(2);
    if (!args.length) { console.error('usage: node tools/convertd_call.js <json> | <cmd> key=value ...'); process.exit(2); }
    let obj;
    if (args[0].startsWith('{')) {
        obj = JSON.parse(args[0]);
    } else {
        obj = { cmd: args[0] };
        for (const a of args.slice(1)) {
            const i = a.indexOf('=');
            if (i < 0) { console.error('bad arg (need key=value): ' + a); process.exit(2); }
            obj[a.slice(0, i)] = a.slice(i + 1);
        }
    }
    const child = spawn(EXE, ['0'], { stdio: ['ignore', 'ignore', 'pipe'] });
    let info = '';
    let sent = false;
    child.stderr.on('data', (d) => {
        info += d.toString();
        const m = info.match(/PORT (\d+)/);
        if (m && !sent) {
            sent = true;
            const s = net.connect(Number(m[1]), '127.0.0.1', () => {
                s.write(JSON.stringify(obj) + '\n');
            });
            let out = '';
            s.on('data', (d) => {
                out += d.toString();
                const idx = out.indexOf('\n');
                if (idx >= 0) {
                    const line = out.slice(0, idx).trim();
                    process.stdout.write(line + '\n');
                    try { process.exit(JSON.parse(line).ok ? 0 : 1); } catch (e) { process.exit(1); }
                }
            });
            s.on('error', (e) => { console.error('connect: ' + e.message); process.exit(1); });
        }
    });
    child.on('error', (e) => { console.error('cannot spawn ' + EXE + ': ' + e.message); process.exit(1); });
}

main();
