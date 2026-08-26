// compare_trace.js - compare two StreamMoE trace dumps (SMT1 binary format)
// Usage: node tools/compare_trace.js <baseline.bin> <expert_backend.bin>
// Reports the first diverging record (name) and per-record max abs diff.
const fs = require('fs');

function readRecords(path) {
    const buf = fs.readFileSync(path);
    const recs = [];
    let off = 0;
    while (off + 4 <= buf.length) {
        const magic = buf.readUInt32LE(off); off += 4;
        if (magic !== 0x31544D53) throw new Error(`bad magic at ${off - 4}`);
        const nameLen = buf.readUInt32LE(off); off += 4;
        const name = buf.toString('utf8', off, off + nameLen); off += nameLen;
        const ne = [buf.readInt32LE(off), buf.readInt32LE(off+4), buf.readInt32LE(off+8), buf.readInt32LE(off+12)]; off += 16;
        const elem = buf.readUInt32LE(off); off += 4;
        const nbytes = buf.readBigUInt64LE(off); off += 8;
        const data = buf.subarray(off, off + Number(nbytes)); off += Number(nbytes);
        recs.push({ name, ne, elem, nbytes, data });
    }
    return recs;
}

const [,, a, b] = process.argv;
if (!a || !b) { console.error('usage: node tools/compare_trace.js <a> <b>'); process.exit(1); }
const A = readRecords(a), B = readRecords(b);
console.log(`records: A=${A.length} B=${B.length}`);

// match records by name (in order); report first divergence
const byName = new Map();
B.forEach((r, i) => { if (!byName.has(r.name)) byName.set(r.name, []); byName.get(r.name).push(i); });

let diverged = 0;
let firstDivergence = null;
for (const rec of A) {
    const cand = (byName.get(rec.name) || []).shift();
    if (cand === undefined) { console.log(`  MISSING in B: ${rec.name}`); diverged++; continue; }
    const rb = B[cand];
    if (rec.nbytes !== rb.nbytes) { console.log(`  SIZE DIFF ${rec.name}: A=${rec.nbytes} B=${rb.nbytes}`); diverged++; if(!firstDivergence) firstDivergence=rec.name; continue; }
    let maxAbs = 0, firstByte = -1;
    for (let i = 0; i < rec.data.length; i++) {
        if (rec.data[i] !== rb.data[i]) {
            const d = Math.abs(rec.data[i] - rb.data[i]);
            if (d > maxAbs) maxAbs = d;
            if (firstByte < 0) firstByte = i;
        }
    }
    if (maxAbs > 0) {
        // interpret as float if 4-byte elem
        let fdiff = 0;
        if (rec.elem === 4) {
            for (let i = 0; i + 4 <= rec.data.length; i += 4) {
                const fa = rec.data.readFloatLE(i), fb = rb.data.readFloatLE(i);
                const d = Math.abs(fa - fb);
                if (d > fdiff) fdiff = d;
            }
        }
        console.log(`  DIFF ${rec.name} ne=[${rec.ne}] bytesDiff=${maxAbs} maxFloatDiff=${fdiff.toExponential(3)}`);
        diverged++;
        if (!firstDivergence) firstDivergence = rec.name;
    }
}
console.log(diverged === 0 ? 'RESULT: IDENTICAL' : `RESULT: ${diverged} record(s) differ; first divergence at: ${firstDivergence}`);
