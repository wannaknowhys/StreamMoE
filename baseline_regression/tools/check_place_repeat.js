// check_place_repeat.js - same-config repeat-consistency gate for place-* engines.
//
// For each place engine: one warm-up run (discarded), then N independent runs.
// Only the first TURNS turns are compared (the B1-on/off and 10-29 races all
// diverged in turn 1), which cuts the cost ~5x. The stripped text must be
// identical across the N runs; a mismatch is a FAIL (a real race). A warm-up
// text that differs from the N runs is only a WARNING (cold-start variants are
// suspected, not confirmed).
//
// Every sample is validated: the output must be the one created after the call
// started, contain `turns` turn blocks, no `[ERROR]`, and enough model lines.
// A crash with a zero exit code must not count as a sample.
//
// Per-engine baseline hash (repeat_baseline.json): the current output hash is
// the reference; any change is a WARNING (the exact text vs CPU is no longer a
// FAIL). The KL adjudication is only for a DIFF that needs a decision.
//
// Usage: node baseline_regression/tools/check_place_repeat.js [--flavor F] [--engines a,b] [--runs 3] [--turns 2] [--update-baseline]

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const REPO = path.resolve(__dirname, '..', '..');
const args = process.argv.slice(2);
let flavor = 'StreamMoE_dump';
let engineFilter = null;
let runs = 3;
let turns = 2;
let updateBaseline = false;
let acceptChanges = false;
let temperature = null;
for (let i = 0; i < args.length; i++) {
    if (args[i] === '--flavor') flavor = args[++i];
    else if (args[i] === '--engines') engineFilter = args[++i].split(',');
    else if (args[i] === '--runs') runs = parseInt(args[++i], 10);
    else if (args[i] === '--turns') turns = parseInt(args[++i], 10);
    else if (args[i] === '--update-baseline') updateBaseline = true;
    else if (args[i] === '--accept') { acceptChanges = true; updateBaseline = true; }
    else if (args[i] === '--temperature') temperature = parseFloat(args[++i]);
}

const enginesDir = path.join(REPO, 'tools', 'run_specs', 'engines');
let engines = fs.readdirSync(enginesDir).filter((f) => f.startsWith('place-') && f.endsWith('.json')).map((f) => f.slice(0, -5));
if (engineFilter) engines = engines.filter((e) => engineFilter.includes(e));
engines.sort();

const baselineFile = path.join(__dirname, 'repeat_baseline.json');
let baseline = {};
if (fs.existsSync(baselineFile)) { try { baseline = JSON.parse(fs.readFileSync(baselineFile, 'utf8')); } catch { baseline = {}; } }

function turnsOf(text) {    const lines = text.split(/\r?\n/);
    const blocks = [];
    let cur = null;
    for (const l of lines) {
        if (/^--- \[Turn/.test(l)) { if (cur) blocks.push(cur); cur = []; continue; }
        if (!cur) continue;
        if (/^(RUN:|DATE:|={5,}|\s*$|pp:|tg:)/.test(l)) continue;
        cur.push(l);
    }
    if (cur) blocks.push(cur);
    return blocks;
}

function validateText(raw, wantTurns) {
    if (raw.includes('[ERROR]')) return { ok: false, err: 'output contains [ERROR]' };
    const blocks = turnsOf(raw);
    if (blocks.length < wantTurns) return { ok: false, err: `only ${blocks.length}/${wantTurns} turn blocks` };
    const modelLines = blocks.slice(0, wantTurns).reduce((n, b) => n + b.length, 0);
    if (modelLines < 20) return { ok: false, err: `too few model lines (${modelLines})` };
    return { ok: true, blocks };
}

let callSeq = 0;

function runOnce(engine, outdir, logdir) {
    fs.mkdirSync(outdir, { recursive: true });
    fs.mkdirSync(logdir, { recursive: true });
    const log = path.join(logdir, 'bench.log');
    const start = Date.now(); const call = ++callSeq;
    const r = spawnSync(process.execPath, [path.join(REPO, 'tools', 'run_bench.js'),
        '--flavors', flavor, '--models', 'olmoe', '--engines', engine, '--tasks', 'en',
        '--runs', '1', '--outdir', outdir, '--max-turns', String(turns)].concat(temperature == null ? [] : ['--temperature', String(temperature)]), { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
    fs.writeFileSync(log, `call=${call} start=${start}\n` + (r.stdout || '').toString() + (r.stderr || '').toString());
    if (r.status !== 0) return { ok: false, err: `run_bench exit ${r.status} (log ${log})` };
    const roots = fs.readdirSync(path.join(REPO, 'temp', 'bench_outputs'))
        .map((d) => path.join(REPO, 'temp', 'bench_outputs', d))
        .filter((d) => fs.statSync(d).isDirectory())
        .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
    // The output must be created after this call started.
    const latest = roots.find((d) => fs.statSync(d).mtimeMs >= start);
    if (!latest) return { ok: false, err: 'no bench output dir created after the call (stale?)' };
    const txt = fs.readdirSync(latest).find((f) => f.includes(engine) && f.endsWith('.txt'));
    if (!txt) return { ok: false, err: 'no engine txt in ' + latest };
    const raw = fs.readFileSync(path.join(latest, txt), 'utf8');
    const v = validateText(raw, turns);
    if (!v.ok) return { ok: false, err: v.err };
    return { ok: true, text: v.blocks.slice(0, turns).map((b) => b.join('\n')).join('\n'), raw };
}

let failures = 0, warnings = 0;
const tag = `${flavor}_${new Date().toISOString().replace(/[:.]/g, '-')}`;
for (const engine of engines) {
    const outdir = path.join(REPO, 'temp', 'repeat_gate', tag, engine);
    const warm = runOnce(engine, path.join(outdir, 'warm'), path.join(outdir, 'warm'));
    const samples = [];
    for (let i = 0; i < runs; i++) samples.push(runOnce(engine, path.join(outdir, `run${i}`), path.join(outdir, `run${i}`)));
    const bad = samples.filter((s) => !s.ok);
    if (bad.length) { console.log(`[FAIL] ${engine}: ${bad[0].err}`); failures++; continue; }
    const same = samples.every((s) => s.text === samples[0].text);
    if (!same) { console.log(`[FAIL] ${engine}: ${runs} runs differ (first ${turns} turns)`); failures++; continue; }
    if (!warm.ok) {
        console.log(`[WARN] ${engine}: warm-up invalid (${warm.err})`);
        warnings++;
    } else if (warm.text !== samples[0].text) {
        console.log(`[WARN] ${engine}: warm-up differs from the stable runs (suspected cold-start variant)`);
        warnings++;
    }
    const key = `${flavor}:${engine}`;
    const h = require('crypto').createHash('sha256').update(samples[0].text).digest('hex').slice(0, 16);
    if (!(key in baseline)) {
        baseline[key] = h;
        console.log(`[NEW]  ${engine}: ${runs} runs identical, baseline recorded ${h} (not a PASS yet)`);
        warnings++;
    } else if (baseline[key] !== h) {
        const old = baseline[key];
        if (acceptChanges) {
            baseline[key] = h;
            console.log(`[ACCEPT] ${engine}: hash ${old} -> ${h} (accepted, update-baseline)`);
        } else {
            console.log(`[WARN] ${engine}: output hash changed ${old} -> ${h} (${runs} runs still identical; use --accept to rebaseline)`);
            warnings++;
        }
    } else {
        console.log(`[PASS] ${engine}: ${runs} runs identical (hash ${h})`);
    }
}
if (updateBaseline) { fs.writeFileSync(baselineFile, JSON.stringify(baseline, null, 2)); console.log(`[baseline] written ${baselineFile}`); }
console.log(failures === 0 ? `RESULT: ${warnings ? `PASS (with ${warnings} warning${warnings === 1 ? '' : 's'})` : 'PASS'} (repeat-consistency, ${engines.length} engines, ${turns} turns)` : `RESULT: FAIL (${failures} engines)`);
process.exit(failures !== 0 ? 1 : (warnings !== 0 ? 2 : 0));



