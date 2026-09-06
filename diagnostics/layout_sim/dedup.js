// Extract clean per-layer life-interval CSV from the raw llama-cli CAP_CSV
// export (which repeats the full dump once per graph build / ubatch).
// Usage: node dedup.js <raw.csv> <out.csv>
const fs = require('fs');
const [, , rawPath, outPath] = process.argv;
if (!rawPath || !outPath) { console.error('usage: node dedup.js <raw> <out>'); process.exit(1); }
const raw = fs.readFileSync(rawPath, 'utf8');
const rows = raw.split('\n').filter(l => /^#layer|^\d+,\d+,/.test(l));
const seen = new Set();
const out = [];
for (const l of rows) {
    if (l.startsWith('#')) { if (!out.includes(l)) out.push(l); continue; }
    const [layer, idx] = l.split(',');
    const k = layer + ':' + idx;
    if (seen.has(k)) continue;   // drop repeated dumps of the same build
    seen.add(k);
    out.push(l);
}
fs.writeFileSync(outPath, out.join('\n') + '\n');
console.log(`rows: ${rows.length} -> dedup ${out.length - 1} (header excluded)`);
