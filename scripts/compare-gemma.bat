@echo off
rem compare gemma (orig vs v1 vs v2) x (upstream vs moe) x (cn vs en) = 12 runs.
rem Needs temp\sm_env.bat with local machine paths - see scripts\sm_env.example.bat.
call "%~dp0..\temp\sm_env.bat" 2>nul
if errorlevel 1 ( echo [compare-gemma] missing temp\sm_env.bat & exit /b 1 )
node tools\run_export.js --models tools\run_specs\models\gemma.json,tools\run_specs\models\gemma-v1.json,tools\run_specs\models\gemma-v2.json --engines tools\run_specs\engines\upstream.json,tools\run_specs\engines\moe.json --tasks tools\run_specs\tasks\cn.json,tools\run_specs\tasks\en.json
