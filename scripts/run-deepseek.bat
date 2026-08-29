@echo off
rem deepseek (upstream vs moe) x (prefill10000, cn, en) = 6 runs.
rem Needs temp\sm_env.bat with local machine paths - see temp\sm_env.example.bat.
call "%~dp0..\temp\sm_env.bat" 2>nul
if errorlevel 1 ( echo [run-deepseek] missing temp\sm_env.bat & exit /b 1 )
node tools\run_export.js --models tools\run_specs\models\deepseek.json --engines tools\run_specs\engines\upstream.json,tools\run_specs\engines\moe.json --tasks tools\run_specs\tasks\prefill10000.json,tools\run_specs\tasks\cn.json,tools\run_specs\tasks\en.json
