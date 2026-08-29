@echo off
pushd %~dp0

call temp\sm_env.bat
node tools\run_export.js --models tools\run_specs\models\deepseek.json --engines tools\run_specs\engines\upstream.json,tools\run_specs\engines\moe.json --tasks tools\run_specs\tasks\prefill10000.json,tools\run_specs\tasks\cn.json,tools\run_specs\tasks\en.json

call temp\sm_env.bat
node tools\run_export.js --models tools\run_specs\models\deepseek-v1.json --engines tools\run_specs\engines\moe.json --tasks tools\run_specs\tasks\prefill10000.json,tools\run_specs\tasks\cn.json,tools\run_specs\tasks\en.json

popd

pause

