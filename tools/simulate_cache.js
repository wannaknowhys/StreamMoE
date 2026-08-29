// simulate_cache.js - expert access history -> cache policy hit-rate curves.
// Reads <LLM_EXPORT_DIR>/expert_history.bin produced by the export build
// (patches/prefill-export-llama.patch), replays the access sequence under
// different cache policies and pool sizes WITHOUT re-running the model.
//
// Format (EXPHIST1): u32 n_recs, then per rec: u32 layer, u32 token, u32 expert.
// Cache key = layer*256 + expert (per-layer expert identity).
//
// Dedup: the same expert is recorded once per MUL_MAT_ID for one token
// (gate/up/down), so (layer, token, expert) is deduplicated before replay -
// otherwise hit rates are inflated by intra-token repeats.
//
// Policies: LRU, LFU, EST1 (recency-weighted decaying counter, simplified).
// Usage: node tools/simulate_cache.js <expert_history.bin> [maxSlots]
const fs = require('fs');

const b = fs.readFileSync(process.argv[2]);
if (b.toString('ascii', 0, 8) !== 'EXPHIST1') throw new Error('bad magic');
const n = b.readUInt32LE(8);
const recs = new Uint32Array(n * 3);
for (let i = 0; i < n * 3; i++) recs[i] = b.readUInt32LE(12 + i * 4);

// dedupe (layer, token, expert), then replay in GLOBAL TOKEN order (one token
// = one step through ALL layers). The raw file is ordered per-layer per-ubatch
// (all tokens of layer 0, then layer 1, ...), so replaying it as-is inflates
// hit rates: each layer fills the pool once and every later access hits.
const seen = new Set();
const entries = [];
for (let i = 0; i < n; i++) {
    const k = recs[i * 3] + '_' + recs[i * 3 + 1] + '_' + recs[i * 3 + 2];
    if (!seen.has(k)) { seen.add(k); entries.push({ t: recs[i * 3 + 1], k: recs[i * 3] * 256 + recs[i * 3 + 2] }); }
}
entries.sort((a, b) => a.t - b.t);
const keys = entries.map((e) => e.k);
const m = keys.length;
const layers = new Set();
for (let i = 0; i < m; i++) layers.add((keys[i] / 256) | 0);
console.log(`records: ${m} (dedup of ${n})  layers:${layers.size} uniqueExperts:${new Set(keys).size}`);

const maxSlots = process.argv[3] ? parseInt(process.argv[3]) : 4096;
const steps = [32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096];
for (let s = 4096 + 256; s <= 11008; s += 256) steps.push(s);
const sizes = [...new Set([...steps, maxSlots].filter(s => s <= 11008 && s >= 1))].sort((a, b) => a - b);

// OPT (Belady): offline - evict the key whose NEXT access is furthest in the future
const optNext = new Int32Array(m);
const optLast = new Map();
for (let i = m - 1; i >= 0; i--) {
    const key = keys[i];
    optNext[i] = optLast.has(key) ? optLast.get(key) : 2147483647;
    optLast.set(key, i);
}
function replayOpt(slots) {
    let hit = 0;
    const cache = new Map();
    for (let i = 0; i < m; i++) {
        const key = keys[i];
        if (cache.has(key)) { hit++; cache.set(key, optNext[i]); }
        else if (cache.size < slots) { cache.set(key, optNext[i]); }
        else {
            let maxK = null, maxP = -1;
            for (const [k, p] of cache) if (p > maxP) { maxP = p; maxK = k; }
            cache.delete(maxK);
            cache.set(key, optNext[i]);
        }
    }
    return hit / m;
}

function replay(policy, slots) {
    let hit = 0;
    const mm = new Map(); // key -> lastAccess (insertion order = recency) or count (LFU)
    const freq = new Map();
    for (let i = 0; i < m; i++) {
        const key = keys[i];
        const has = mm.has(key);
        if (has) hit++;
        if (policy === 'lru') {
            if (has) { mm.delete(key); mm.set(key, i); }
            else if (mm.size < slots) { mm.set(key, i); }
            else { const oldest = mm.keys().next().value; mm.delete(oldest); mm.set(key, i); }
        } else if (policy === 'lfu') {
            if (!has && mm.size >= slots) {
                let minK = null, minC = Infinity;
                for (const [k, v] of mm) if (v < minC) { minC = v; minK = k; }
                mm.delete(minK); freq.delete(minK);
            }
            if (mm.size < slots) { mm.set(key, (freq.get(key) || 0) + 1); }
        } else { // est1: recency-weighted decaying counter (score = count, halves every 64 accesses)
            if (has) { freq.set(key, (freq.get(key) || 0) + 1); mm.delete(key); mm.set(key, i); }
            else if (mm.size < slots) { mm.set(key, i); freq.set(key, 1); }
            else {
                let minK = null, minS = Infinity;
                for (const [k, v] of mm) {
                    const c = (freq.get(k) || 0) / (Math.pow(2, (i - v) / 64) + 1e-30);
                    if (c < minS) { minS = c; minK = k; }
                }
                mm.delete(minK); freq.delete(minK); mm.set(key, i); freq.set(key, 1);
            }
        }
    }
    return hit / m;
}

console.log('\npool_slots  LRU       LFU       EST1      OPT');
for (const s of sizes) {
    const r = [replay('lru', s), replay('lfu', s), replay('est1', s), replayOpt(s)].map(v => v.toFixed(4));
    console.log(`${String(s).padStart(10)}  ${r.join('   ')}`);
}
