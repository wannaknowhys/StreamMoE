// run_export_batch.js - run prefill cross-validation + expert-history export over a jsonl prompt set.
// For each prompt: std run (no --expert-backend) -> temp/export_<lang>_<i>_std, moe run
// (--expert-backend) -> temp/export_<lang>_<i>_moe, then verify_prefill.js and simulate_cache.js.
// Usage: node tools/run_export_batch.js [en|zh] [n_prompts]  (default en, 1)
const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const lang = process.argv[2] || 'en';
const nLimit = process.argv[3] ? parseInt(process.argv[3], 10) : 1;
const model = 'N:\\AI_LLM\\DeepSeek-V4-Flash-0731\\DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf';
const exe = 'build\\main\\bin\\stream_moe.exe';
const jsonl = lang === 'zh' ? 'benchmark\\prompts\\long_horizon_prompts_zh.jsonl' : 'benchmark\\prompts\\long_horizon_prompts.jsonl';

const raw = fs.readFileSync(jsonl, 'utf8').replace(/^\uFEFF/, ''); // strip UTF-8 BOM
const lines = raw.split(/\r?\n/).filter((l) => l.trim());
const prompts = lines.map((l) => JSON.parse(l).prompt).slice(0, nLimit);
console.log(`[run_export_batch] lang=${lang} jsonl=${jsonl} prompts=${prompts.length}`);

function run(prompt, extra, dir) {
    fs.mkdirSync(dir, { recursive: true });
    for (const f of ['prefill_export.bin', 'expert_history.bin']) {
        const p = path.join(dir, f);
        if (fs.existsSync(p)) fs.unlinkSync(p);
    }
    const r = spawnSync(exe, ['-m', model, '--moe-ram-pool', '71680', '--temp', '0', '-c', '4096', '-t', '16', '-p', prompt, '-n', '2', ...extra], {
        env: { ...process.env, OMP_NUM_THREADS: '16', LLM_EXPORT_DIR: dir },
        encoding: 'utf8', maxBuffer: 16 * 1024 * 1024, timeout: 30 * 60 * 1000,
    });
    return r;
}

for (let i = 0; i < prompts.length; i++) {
    const brief = prompts[i].replace(/\n/g, ' ').slice(0, 50);
    console.log(`\n=== [${lang}] prompt ${i + 1}/${prompts.length}: ${brief}... ===`);
    const stdDir = path.join('temp', `export_${lang}_${i}_std`);
    const moeDir = path.join('temp', `export_${lang}_${i}_moe`);

    console.log('  [std] running ...');
    const rs = run(prompts[i], [], stdDir);
    console.log(`  [std] exit=${rs.status} ${rs.error ? '(timeout/err)' : ''}`);
    if (rs.status !== 0) { console.log('  [std] stderr tail:\n' + (rs.stderr || '').slice(-800)); }

    console.log('  [moe] running ...');
    const rm = run(prompts[i], ['--expert-backend'], moeDir);
    console.log(`  [moe] exit=${rm.status} ${rm.error ? '(timeout/err)' : ''}`);
    if (rm.status !== 0) { console.log('  [moe] stderr tail:\n' + (rm.stderr || '').slice(-800)); }

    const pStd = path.join(stdDir, 'prefill_export.bin');
    const pMoe = path.join(moeDir, 'prefill_export.bin');
    if (fs.existsSync(pStd) && fs.existsSync(pMoe)) {
        const v = spawnSync('node', ['tools/verify_prefill.js', pStd, pMoe], { encoding: 'utf8' });
        console.log('  verify_prefill:\n  ' + (v.stdout || '').trim().split('\n').join('\n  '));
    } else {
        console.log('  verify_prefill: SKIP (missing export file)');
    }
    const pHist = path.join(moeDir, 'expert_history.bin');
    if (fs.existsSync(pHist)) {
        const s = spawnSync('node', ['tools/simulate_cache.js', pHist, '11008'], { encoding: 'utf8' });
        console.log('  simulate_cache:\n  ' + (s.stdout || '').trim().split('\n').join('\n  '));
    } else {
        console.log('  simulate_cache: SKIP (no expert_history.bin)');
    }
}
console.log('\n[done] exports under temp/export_<lang>_<i>_{std,moe}');
