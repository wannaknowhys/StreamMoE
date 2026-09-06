// Layout simulator - compare result-buffer allocators on real per-layer
// life-interval data (diagnostics/layout_sim/*_life.csv).
//
// Each compute node i has a live interval [start=i, end=last_use[i]] (it is
// written by node i, read by last_use, dead after). It needs `bytes`. Two
// results may share an address iff their intervals do not overlap. We compare
// the total per-layer buffer (peak address) each strategy needs.
//
// Strategies:
//   firstfit   : current C++ greedy - exec order, reuse first slot whose
//                occupant died (last_use < current index), slot sized to max.
//   bestfitdec : sort by size desc, place each at lowest offset whose live
//                occupants never overlap its interval.
//   compact2   : simulated malloc/free over a linear arena + compaction
//                (user's idea): free dead blocks, first-fit into a gap, else
//                re-pack live blocks to the front and place. Move cost is zero
//                at layout time (data is not written yet).
//   lower      : theoretical lower bound = max over time of sum(bytes) of
//                simultaneously-live results (peak live demand).
//
// Usage: node sim.js <csv...>
const fs = require('fs');

function load(path) {
    const layers = new Map();
    for (const l of fs.readFileSync(path, 'utf8').split('\n')) {
        if (!/^\d+,\d+,/.test(l)) continue;
        const [ly, i, name, op, bytes, last] = l.split(',');
        if (!layers.has(ly)) layers.set(ly, []);
        layers.get(ly).push({ i: +i, name, bytes: +bytes, last: +last });
    }
    for (const arr of layers.values()) arr.sort((a, b) => a.i - b.i);
    return layers;
}

// ---- current C++ greedy (first-fit slot, slot sized to max occupant) ------
function firstfit(nodes) {
    const C = nodes.length;
    const slots = [];
    for (let i = 0; i < C; ++i) {
        const nb = nodes[i].bytes;
        const end = nodes[i].last < 0 ? C : nodes[i].last + 1;
        let ch = -1;
        for (let s = 0; s < slots.length; ++s)
            if (slots[s].end <= i) { ch = s; break; }
        if (ch < 0) { ch = slots.length; slots.push({ size: 0, end: 0 }); }
        slots[ch].size = Math.max(slots[ch].size, nb);
        slots[ch].end = Math.max(slots[ch].end, end);
    }
    return slots.reduce((a, s) => a + s.size, 0);
}

// ---- best-fit decreasing by node size -------------------------------------
function bestfitdec2(nodes) {
    const order = nodes.map((_, i) => i).sort((a, b) => nodes[b].bytes - nodes[a].bytes);
    const placed = [];
    let arena = 0;
    for (const ni of order) {
        const n = nodes[ni];
        const start = n.i, end = n.last < 0 ? Infinity : n.last;
        let off = 0;
        for (;;) {
            const blockers = placed.filter(p =>
                !(end < p.start || p.end < start) &&      // time-overlap
                off < p.off + p.bytes && p.off < off + n.bytes);
            if (blockers.length === 0) break;
            off = Math.max(...blockers.map(b => b.off + b.bytes));  // jump past
        }
        placed.push({ off, bytes: n.bytes, start, end });
        arena = Math.max(arena, off + n.bytes);
    }
    return arena;
}

// ---- simulated malloc/free + compaction (user's idea) ---------------------
function compact2(nodes) {
    let live = [];       // {off,size,start,end}
    let tail = 0;
    for (let i = 0; i < nodes.length; ++i) {
        const n = nodes[i];
        live = live.filter(b => b.end >= i);            // free dead blocks
        const end = n.last < 0 ? Infinity : n.last;
        let cur = 0;
        for (const b of live) cur = Math.max(cur, b.off + b.size);   // live tail
        let off = placeInGaps(live, cur, n.bytes);
        if (off === null) {
            // compaction: re-pack live blocks to the front (layout-time free)
            const packed = [];
            let c2 = 0;
            for (const b of live) { packed.push({ ...b, off: c2 }); c2 += b.size; }
            live = packed;
            cur = c2;
            off = placeInGaps(live, cur, n.bytes);
            if (off === null) { off = cur; cur += n.bytes; }   // no room: append
        }
        live.push({ off, size: n.bytes, start: i, end });
        tail = Math.max(tail, off + n.bytes);
    }
    return tail;
    function placeInGaps(lv, occ, need) {
        const seg = lv.slice().sort((a, b) => a.off - b.off);
        let cursor = 0;
        for (const b of seg) {
            if (b.off > cursor && b.off - cursor >= need) return cursor;
            cursor = Math.max(cursor, b.off + b.size);
        }
        if (occ - cursor >= need) return cursor;
        return null;
    }
}

// ---- theoretical lower bound ----------------------------------------------
function lowerBound(nodes) {
    const events = [];
    for (const n of nodes) {
        const end = n.last < 0 ? nodes.length - 1 : n.last;
        events.push([n.i, +n.bytes]);
        events.push([end + 1, -n.bytes]);
    }
    events.sort((a, b) => a[0] - b[0]);
    let cur = 0, peak = 0;
    for (const [, d] of events) { cur += d; if (cur > peak) peak = cur; }
    return peak;
}

const ALGS = {
    firstfit:   { f: firstfit,   desc: 'first-fit slot (current C++)' },
    bestfitdec: { f: bestfitdec2, desc: 'best-fit decreasing by size' },
    compact2:   { f: compact2,   desc: 'sim malloc/free + compaction' },
    lower:      { f: lowerBound, desc: 'lower bound (peak live bytes)' },
};

function main() {
    const files = process.argv.slice(2);
    if (!files.length) { console.error('usage: node sim.js <csv...>'); process.exit(1); }
    for (const f of files) {
        const layers = load(f);
        console.log(`\n=== ${f} : ${layers.size} layers ===`);
        const aggr = {};
        for (const k of Object.keys(ALGS)) aggr[k] = { sum: 0, max: 0 };
        const keys = [...layers.keys()].sort((a, b) => a - b);
        console.log('layer | ' + Object.entries(ALGS).map(([k]) => k.padStart(11)).join(' | '));
        for (const ly of keys) {
            const nodes = layers.get(ly);
            const row = {};
            for (const [k, v] of Object.entries(ALGS)) {
                const r = v.f(nodes);
                row[k] = r;
                aggr[k].sum += r;
                aggr[k].max = Math.max(aggr[k].max, r);
            }
            if (keys.length <= 6 || ly === keys[0] || ly === keys[keys.length - 1])
                console.log(ly.padStart(5) + ' | ' + Object.values(row).map(x => String(x).padStart(11)).join(' | '));
        }
        console.log('TOTAL | ' + Object.entries(ALGS).map(([k]) => String(aggr[k].sum).padStart(11)).join(' | '));
        const ff = aggr.firstfit.sum, lb = aggr.lower.sum;
        console.log(`vs firstfit: bestfitdec ${(100 * aggr.bestfitdec.sum / ff).toFixed(0)}%  compact ${(100 * aggr.compact2.sum / ff).toFixed(0)}%  lower ${(100 * lb / ff).toFixed(0)}%`);
    }
}
main();
