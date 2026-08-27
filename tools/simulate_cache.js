// simulate_cache.js - expert access history -> cache policy hit-rate curves.
// Reads <LLM_EXPORT_DIR>/expert_history.bin produced by the export build
// (patches/prefill-export-llama.patch), replays the access sequence under
// different cache policies and pool sizes WITHOUT re-running the model.
//
// Format (EXPHIST1): u32 n_recs, then per rec: u32 layer, u32 token, u32 expert.
// Cache key = layer*256 + expert (per-layer expert identity).
//
// Policies: LRU, LFU, EST1 (recency-weighted decaying counter, simplified).
// Usage: node tools/simulate_cache.js <expert_history.bin> [maxSlots]
const fs = require('fs');

const b = fs.readFileSync(process.argv[2]);
if (b.toString('ascii', 0, 8) !== 'EXPHIST1') throw new Error('bad magic');
const n = b.readUInt32LE(8);
const recs = new Uint32Array(n * 3);
for (let i = 0; i < n * 3; i++) recs[i] = b.readUInt32LE(12 + i * 4);
console.log(`records: ${n}  layers:${[...new Set(Array.from({length:n},(_,i)=>recs[i*3]))].length} uniqueExperts:${new Set(Array.from({length:n},(_,i)=>recs[i*3]*256+recs[i*3+2])).size}`);

const maxSlots = process.argv[3] ? parseInt(process.argv[3]) : 4096;
const steps = [32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096];
const sizes = [...new Set([...steps, maxSlots].filter(s => s <= 11008 && s >= 1))].sort((a, b) => a - b);

function replay(policy, slots) {
    let hit = 0;
    // LRU/EST1: Map key->lastAccess (insertion order = recency). LFU: Map key->count.
    const m = new Map();
    const freq = new Map();
    for (let i = 0; i < n; i++) {
        const key = recs[i * 3] * 256 + recs[i * 3 + 2];
        const has = m.has(key);
        if (has) hit++;
        if (policy === 'lru') {
            if (has) { m.delete(key); m.set(key, i); }
            else if (m.size < slots) { m.set(key, i); }
            else { const oldest = m.keys().next().value; m.delete(oldest); m.set(key, i); }
        } else if (policy === 'lfu') {
            if (!has && m.size >= slots) {
                let minK = null, minC = Infinity;
                for (const [k, v] of m) if (v < minC) { minC = v; minK = k; }
                m.delete(minK); freq.delete(minK);
            }
            if (m.size < slots) { m.set(key, (freq.get(key) || 0) + 1); }
        } else { // est1: recency-weighted decaying counter (score = count, halves every 64 accesses)
            if (has) { freq.set(key, (freq.get(key) || 0) + 1); m.delete(key); m.set(key, i); }
            else if (m.size < slots) { m.set(key, i); freq.set(key, 1); }
            else {
                const t = i / 64;
                let minK = null, minS = Infinity;
                for (const [k, v] of m) {
                    const c = (freq.get(k) || 0) / (Math.pow(2, (i - v) / 64) + 1e-30);
                    if (c < minS) { minS = c; minK = k; }
                }
                m.delete(minK); freq.delete(minK); m.set(key, i); freq.set(key, 1);
            }
        }
    }
    return hit / n;
}

console.log('\npool_slots  LRU       LFU       EST1');
for (const s of sizes) {
    const r = [replay('lru', s), replay('lfu', s), replay('est1', s)].map(v => v.toFixed(4));
    console.log(`${String(s).padStart(10)}  ${r.join('   ')}`);
}
