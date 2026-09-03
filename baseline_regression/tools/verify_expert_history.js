// verify_expert_history.js - compare two EXPHIST1 files (per-token expert
// routing history, exported to <LLM_EXPORT_DIR>/expert_history_main.bin).
// Format: "EXPHIST1" + u32 n + n x (u32 layer, u32 token, u32 expert_id).
// Reports entry counts, first diverging entry, total diffs, size delta.
// usage: node tools/verify_expert_history.js <std.bin> <moe.bin>
const fs = require('fs');
const [a, b] = process.argv.slice(2);
if (!a || !b) { console.error('usage: node tools/verify_expert_history.js <f1> <f2>'); process.exit(2); }
const pa = fs.readFileSync(a), pb = fs.readFileSync(b);
if (pa.toString('ascii', 0, 8) !== 'EXPHIST1' || pb.toString('ascii', 0, 8) !== 'EXPHIST1') { console.error('bad magic'); process.exit(1); }
const na = pa.readUInt32LE(8), nb = pb.readUInt32LE(8);
console.log(`entries: std=${na} moe=${nb}  sizes: ${pa.length} vs ${pb.length} (delta ${pb.length - pa.length})`);
if (na !== nb) console.log('  NOTE: entry count differs - routing history is not aligned');

const N = Math.min(na, nb);
let diff = 0, first = -1;
const item = (buf, i) => { const o = 12 + i * 12; return [buf.readUInt32LE(o), buf.readUInt32LE(o + 4), buf.readUInt32LE(o + 8)]; };
for (let i = 0; i < N; i++) {
    const [l1, t1, x1] = item(pa, i), [l2, t2, x2] = item(pb, i);
    if (l1 !== l2 || t1 !== t2 || x1 !== x2) {
        if (diff < 5) console.log(`  DIFF #${i}: std(layer=${l1},token=${t1},expert=${x1})  moe(layer=${l2},token=${t2},expert=${x2})`);
        if (first < 0) first = i;
        diff++;
    }
}
if (diff === 0) console.log('\nRESULT: IDENTICAL (all expert-history entries match)');
else console.log(`\nRESULT: ${diff}/${N} entries differ, first at #${first}`);
process.exit(diff ? 1 : 0);
