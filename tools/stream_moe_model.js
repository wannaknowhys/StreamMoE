// stream_moe_model.js - converter intermediate format (in-memory model
// abstraction, docs/STREAMMOE_GGUF_FORMAT.md §9). Every GGUF flavor (original /
// v1 / v2 / v2-chunk) parses into the same shape; writers then emit any target
// from it. Lets the converter be verified by round-tripping (A->mid->B == A->B).
//
// Intermediate shape (not a file - metadata + byte-range locators only):
//   { arch, n_layer, n_expert, n_expert_used, layout,
//     dense:  [ { name, ne, type, size, src:{file,off,size} } ],
//     expert: [ { name, ne, type, perExpert, branch, layer,
//                 perExpertSrc:[ {file,off,size} x n_expert ] } ] }
//   perExpertSrc[e] = the exact byte range of expert e for that branch tensor.
//
// usage: node tools/stream_moe_model.js open <model.gguf>
//        (spawns temp/stream_moe_convertd.exe for metadata, parses, validates)
const { spawn } = require('child_process');
const path = require('path');

const EXE = path.join(__dirname, '..', 'temp', 'stream_moe_convertd.exe');

function openMeta(ggufPath) {
    return new Promise((resolve, reject) => {
        const child = spawn(EXE, [], { stdio: ['pipe', 'pipe', 'pipe'] });
        let out = '';
        child.stdout.on('data', (d) => (out += d));
        child.stderr.on('data', (d) => process.stderr.write(d));
        child.on('close', () => { try { resolve(JSON.parse(out)); } catch (e) { reject(new Error('bad wrapper JSON: ' + out.slice(0, 200))); } });
        child.on('error', (e) => reject(new Error('cannot spawn ' + EXE)));
        child.stdin.write(JSON.stringify({ cmd: 'open', path: ggufPath }) + '\n');
        child.stdin.end();
    });
}

const kv = (m, k) => { const f = m.kv.find((x) => x.k === k); return f ? f.v : undefined; };

class Model {
    constructor() { this.arch = ''; this.n_layer = 0; this.n_expert = 0; this.n_expert_used = 0; this.layout = 'original'; this.dense = []; this.expert = []; }

    // original / v1: tensor-info lists the real tensors; experts are contiguous
    // per-tensor (expert e at off + e*perExpert).
    static fromOriginal(m, dataOffset) {
        const mdl = new Model();
        mdl.layout = 'original';
        mdl.arch = kv(m, 'general.architecture') || '';
        mdl.n_layer = Number(kv(m, mdl.arch + '.block_count') || kv(m, 'block_count') || 0);
        mdl.n_expert = Number(kv(m, mdl.arch + '.expert_count') || 0);
        mdl.n_expert_used = Number(kv(m, mdl.arch + '.expert_used_count') || 0);
        for (const t of m.tensors) {
            const isExp = t.name.includes('_exps') && !t.name.includes('.scale');
            if (isExp) {
                const perExpert = t.size / mdl.n_expert;
                const srcs = [];
                for (let e = 0; e < mdl.n_expert; e++) srcs.push({ file: m.path || '', off: dataOffset + t.offset + e * perExpert, size: perExpert });
                mdl.expert.push({ name: t.name, ne: t.ne, type: t.type, perExpert, branch: branchOf(t.name), layer: layerOf(t.name), perExpertSrc: srcs });
            } else {
                mdl.dense.push({ name: t.name, ne: t.ne, type: t.type, size: t.size, src: { file: m.path || '', off: dataOffset + t.offset, size: t.size } });
            }
        }
        return mdl;
    }

    // v2: experts live in compact blocks (one per expert); branch layout comes
    // from expert_branch_names/sizes/counts. perExpertSrc[e] = block e's branch
    // interval - this is the reversible key.
    static fromV2(m, dataOffset) {
        const mdl = new Model();
        mdl.layout = 'v2';
        mdl.arch = kv(m, 'general.architecture') || '';
        mdl.n_layer = Number(kv(m, mdl.arch + '.block_count') || 0);
        mdl.n_expert = Number(kv(m, mdl.arch + '.expert_count') || 0);
        mdl.n_expert_used = Number(kv(m, mdl.arch + '.expert_used_count') || 0);
        const sec = m.kv.find((x) => x.k === 'stream_moe.expert_sections');
        const bnames = m.kv.find((x) => x.k === 'stream_moe.expert_branch_names');
        const bsizes = m.kv.find((x) => x.k === 'stream_moe.expert_branch_sizes');
        const bcounts = m.kv.find((x) => x.k === 'stream_moe.expert_branch_counts');
        if (!sec || !bnames || !bsizes || !bcounts) throw new Error('v2: missing stream_moe.* layout KV');
        const sections = sec.arr || sec.v || [];
        const names = bnames.arr || bnames.v || [];
        const sizes = bsizes.arr || bsizes.v || [];
        const counts = bcounts.arr || bcounts.v || [];
        const nBlocks = sections.length / 3;
        // per-layer branch name + per-expert size, in block order
        const layerBranches = [];
        let nameIdx = 0, sizeIdx = 0;
        for (let l = 0; l < mdl.n_layer; l++) {
            const cnt = counts[l] || 0;
            const bs = [];
            for (let j = 0; j < cnt; j++) { bs.push({ name: names[nameIdx++], perExpert: sizes[sizeIdx++] }); }
            layerBranches.push(bs);
        }
        // dense tensors (real, in dense_section)
        for (const t of m.tensors) {
            if (t.name.includes('_exps') && !t.name.includes('.scale')) continue;
            mdl.dense.push({ name: t.name, ne: t.ne, type: t.type, size: t.size, src: { file: m.path || '', off: dataOffset + t.offset, size: t.size } });
        }
        // per-expert branch srcs from the blocks
        for (let l = 0; l < mdl.n_layer; l++) {
            let branchOff = 0;
            for (const br of layerBranches[l]) {
                const srcs = [];
                for (let e = 0; e < mdl.n_expert; e++) {
                    const blk = l * mdl.n_expert + e;
                    srcs.push({ file: m.path || '', off: dataOffset + sections[blk * 3] + branchOff, size: br.perExpert });
                }
                mdl.expert.push({ name: br.name, ne: null, type: 0, perExpert: br.perExpert, branch: branchOf(br.name), layer: l, perExpertSrc: srcs });
                branchOff += br.perExpert;
            }
        }
        return mdl;
    }

    validate() {
        const errs = [];
        for (const t of this.expert) {
            if (t.perExpertSrc.length !== this.n_expert) errs.push(`${t.name}: perExpertSrc len ${t.perExpertSrc.length} != n_expert ${this.n_expert}`);
            for (const s of t.perExpertSrc) if (s.size !== t.perExpert) errs.push(`${t.name}: expert src size ${s.size} != perExpert ${t.perExpert}`);
        }
        // total bytes: sum dense + expert data == model byte coverage sanity
        const expBytes = this.expert.reduce((a, t) => a + t.perExpert * t.perExpertSrc.length, 0);
        return { errs, denseBytes: this.dense.reduce((a, t) => a + t.size, 0), expertBytes: expBytes, experts: this.expert.length };
    }

    summary() {
        const v = this.validate();
        const branches = {};
        for (const t of this.expert) branches[t.branch] = (branches[t.branch] || 0) + 1;
        return {
            arch: this.arch, layout: this.layout, n_layer: this.n_layer, n_expert: this.n_expert,
            dense: this.dense.length, expertTensors: this.expert.length, branches,
            denseBytes: v.denseBytes, expertBytes: v.expertBytes, errors: v.errs,
        };
    }
}

function branchOf(name) {
    if (name.includes('gate_up')) return 'gate_up';
    if (name.includes('ffn_gate_exps')) return 'gate';
    if (name.includes('ffn_up_exps')) return 'up';
    if (name.includes('down_exps')) return name.includes('.scale') ? 'scale' : 'down';
    return name.split('.').pop() || name;
}
function layerOf(name) {
    const p = name.indexOf('blk.');
    if (p < 0) return -1;
    return parseInt(name.slice(p + 4), 10);
}

async function main() {
    const [cmd, path] = process.argv.slice(2);
    if (cmd !== 'open' || !path) { console.error('usage: node tools/stream_moe_model.js open <model.gguf>'); process.exit(1); }
    const m = await openMeta(path);
    if (!m.ok) { console.error('open failed: ' + m.error); process.exit(1); }
    const meta = m.meta;
    meta.path = path;
    const layout = kv(meta, 'stream_moe.layout') || 'original';
    const mdl = layout === 'expert-blocks-v2' ? Model.fromV2(meta, meta.data_offset) : Model.fromOriginal(meta, meta.data_offset);
    mdl.layout = layout === 'sections-v1' ? 'v1' : mdl.layout;
    console.log(JSON.stringify(mdl.summary(), null, 2));
    if (mdl.validate().errs.length) { console.error('VALIDATION ERRORS:\n' + mdl.validate().errs.slice(0, 10).join('\n')); process.exit(1); }
}

main().catch((e) => { console.error('model: ' + e.message); process.exit(1); });
