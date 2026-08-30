// cmp_gguf.js - byte-compare two files (streaming, no fc). Exit 0 iff identical.
// usage: node tools/cmp_gguf.js <f1> <f2>
const fs = require('fs');
const [a, b] = process.argv.slice(2);
if (!a || !b) { console.error('usage: node tools/cmp_gguf.js <f1> <f2>'); process.exit(2); }
const BUF = 1 << 22;
const fa = fs.openSync(a, 'r'), fb = fs.openSync(b, 'r');
const sizeA = fs.fstatSync(fa).size, sizeB = fs.fstatSync(fb).size;
const ba = Buffer.alloc(BUF), bb = Buffer.alloc(BUF);
let off = 0, first = -1;
while (true) {
    const na = fs.readSync(fa, ba, 0, BUF, off);
    const nb = fs.readSync(fb, bb, 0, BUF, off);
    if (na !== nb) { first = off; break; }
    for (let i = 0; i < na; i++) if (ba[i] !== bb[i]) { first = off + i; break; }
    if (first >= 0) break;
    if (na < BUF) break;
    off += na;
}
if (first < 0) {
    if (sizeA === sizeB) { console.log('  [OK] identical (' + sizeA + ' bytes)'); process.exit(0); }
    first = Math.min(sizeA, sizeB);
}
console.log(`  [DIFF] offset ${first} (0x${first.toString(16)})  sizeA=${sizeA} sizeB=${sizeB}`);
process.exit(1);
