// stream_moe_model.js - Model verification tool. Opens any GGUF source
// (official multi-shard / original / v1 / v2 / v2-chunk), builds the universal
// Model description (tools/stream_moe_layout.js) and prints a summary + self-
// consistency validation. All conversion logic lives in stream_moe_layout.js;
// this is purely an assertion/debug consumer.
//
// usage: node tools/stream_moe_model.js open <model.gguf> [more files...]
const { openFiles, buildModel } = require('./stream_moe_layout');

async function main() {
    const paths = process.argv.slice(2).filter((p) => p !== 'open');
    if (!paths.length) { console.error('usage: node tools/stream_moe_model.js open <model.gguf> [more...]'); process.exit(1); }
    const files = await openFiles(paths);
    const model = buildModel(files);

    const branches = {};
    for (const t of model.expert) branches[t.branch] = (branches[t.branch] || 0) + 1;
    const denseBytes = model.dense.reduce((a, t) => a + t.size, 0);
    const expertBytes = model.expert.reduce((a, t) => a + t.size, 0);
    const errs = [];
    for (const t of model.expert) {
        if (t.perExpertSrcs.length !== model.nExpert) errs.push(`${t.name}: perExpertSrcs ${t.perExpertSrcs.length} != n_expert ${model.nExpert}`);
        for (const segs of t.perExpertSrcs) {
            const tot = segs.reduce((a, s) => a + s.len, 0);
            if (tot !== t.perExpert) errs.push(`${t.name}: expert slice bytes ${tot} != perExpert ${t.perExpert}`);
        }
    }
    for (const t of model.dense) {
        const tot = t.srcs.reduce((a, s) => a + s.len, 0);
        if (tot !== t.size) errs.push(`${t.name}: dense src bytes ${tot} != size ${t.size}`);
    }

    console.log(JSON.stringify({
        arch: model.arch, layout: model.layout, incomplete: model.incomplete,
        nLayer: model.nLayer, nExpert: model.nExpert, nExpertUsed: model.nExpertUsed,
        shards: files.length, dense: model.dense.length, expertTensors: model.expert.length,
        branches, denseBytes, expertBytes, errors: errs.slice(0, 5),
    }, null, 2));
    if (errs.length) { console.error(`VALIDATION ERRORS: ${errs.length}`); process.exit(1); }
}

main().catch((e) => { console.error('model: ' + e.message); process.exit(1); });
