// Regenerate BENCHMARK_REPORT_REAL_<tag>.md from profile_real_<tag>.jsonl
// (mirrors tools/bench_agent.js generateReport output shape)
const fs = require('fs');
const path = require('path');

const tag = process.argv[2] || 'en3';
const profilePath = path.join(__dirname, '..', 'benchmark', 'results', `profile_real_${tag}.jsonl`);
const outPath = path.join(__dirname, '..', 'benchmark', 'results', `BENCHMARK_REPORT_REAL_${tag}.md`);

if (!fs.existsSync(profilePath)) { console.error('missing', profilePath); process.exit(1); }

const lines = fs.readFileSync(profilePath, 'utf8').trim().split('\n');
const resp = lines.map(l => { try { return JSON.parse(l); } catch { return null; } })
  .filter(x => x && x.event === 'response_finish');

let md = `# StreamMoE Long-Horizon Benchmark & Profiling Report\n\n`;
md += `> **Model**: deepseek4 (43 layers, 256 experts/layer)\n`;
md += `> **RAM Pool**: 71680 MB (5621 slots, mmap page-cache baseline)\n`;
md += `> **Runtime State**: REAL_INFERENCE_LIBLLAMA\n\n`;
md += `## 1. Per-Turn Telemetry Summary\n\n`;
md += `| Turn | Prompt Tok | Gen Tok | Prefill TPS | Decode TPS | Total TPS (wall) | Truncated |\n`;
md += `| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n`;

resp.forEach(r => {
  const wall = r.total_duration_ms > 0 ? (r.generated_tokens / (r.total_duration_ms / 1000)).toFixed(1) : '0';
  md += `| **${r.turn}** | ${r.prompt_tokens} | ${r.generated_tokens} | ${r.prefill_tps.toFixed(1)} | ${r.decode_tps.toFixed(1)} | ${wall} | ${r.truncated ? 'yes' : 'no'} |\n`;
});

md += `\n## 2. Technical Findings\n\n`;
md += `1. **Real inference baseline**: libllama deepseek4 forward (MLA/DSA/HC), mmap page-cache expert streaming.\n`;
md += `2. **Speed bound by model drive (N: USB-NVMe) cold page faults**; decode TPS improves as page cache warms.\n`;
md += `3. **Next**: route B custom backend (expert pool, no mmap for MoE) - see docs/LLAMA_MOE_NO_MMAP_RESEARCH.md.\n\n`;

fs.writeFileSync(outPath, md, 'utf8');
console.log('wrote', outPath);
