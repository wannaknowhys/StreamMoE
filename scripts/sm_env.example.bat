@echo off
rem Local machine paths for run_export / run_bench / convert. COPY this to
rem temp\sm_env.bat (or private\env.bat) - those dirs are gitignored, so the
rem local file with real paths is NEVER committed. Fill in your paths after '='.
set SM_GEMMA_ORIG=
set SM_GEMMA_V2=
set SM_GEMMA_V1=
set SM_DS_MAIN=
set SM_DS_DRAFT=
set SM_DS_V1=
set SM_OLMOE=
set SM_JSONL_CN=
set SM_JSONL_EN=
rem JSON array of chat messages used by feed.type=prefill (e.g. a ~10k-token snapshot).
set SM_PREFILL_10K=
set SM_OUT_ROOT=
rem Stock upstream llama.cpp llama-server.exe (Vulkan build), for run_bench stock-* engines.
set SM_UPSTREAM_BIN=
