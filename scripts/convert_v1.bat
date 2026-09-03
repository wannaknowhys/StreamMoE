@echo off
pushd "%~dp0\.."
rem =====================================================================
rem  Convert a GGUF model to the StreamMoE v1 (sections-v1) format.
rem  Front-end: node tools/stream_moe_convert.js -> C++ wrapper
rem  (stream_moe_convertd) -> ggml gguf library.
rem
rem  usage:
rem    scripts\convert_v1.bat -m <model.gguf> -o <out.gguf> [--format v1]
rem
rem  parameters:
rem    -m, --model    model path. For multi-shard models (e.g. deepseek
rem                   ...-00001-of-00005.gguf) pass ANY shard; the other
rem                   shards are auto-discovered via split.count.
rem    -o, --output   output GGUF path (v1 format written here).
rem    --format       v1 (default). v2 (expert-blocks) not implemented yet.
rem
rem  v1 output = GGUF superset: alignment=4096, DENSE section (non-_exps,
rem  source order) then EXPERT section (_exps, source order), every tensor
rem  4K-aligned, stream_moe.* KV. Upstream llama.cpp can load it directly.
rem
rem  example:
rem    scripts\convert_v1.bat -m N:\AI_LLM\gemma-4-26B-A4B-it-UD-Q4_K_M-v2.gguf -o out\gemma_v1.gguf
rem =====================================================================
node tools\stream_moe_convert.js %*
popd
