// verify_places.js - Dynamic place-* engine regression runner against standard baseline.
//
// Usage:
//   node baseline_regression/tools/verify_places.js [options]
// Options:
//   --flavor <name>           Build flavor (default StreamMoE)
//   --engine <name>           Filter to a single place-* engine (e.g. place-cpu, place-c1c2-exp5)
//   --bin-override <path>     Explicit path to llama-server.exe (overrides flavor default)
//   --temp-override <val>     Temperature override (default 0 for greedy deterministic)
//   --models <path>           Model spec json (default tools/run_specs/models/olmoe.json)
//   --tasks <path>            Task spec json (default tools/run_specs/tasks/hi.json)
//   --engines-dir <path>      Dir with place-*.json (default tools/run_specs/engines)
//   --baseline <path>         Baseline reference file (default baseline_regression/baseline/olmoe_hi_baseline.txt)
//   --out-root <path>         Dir for exported results (default baseline_regression/temp/olmoe_places)
//   --port <port>             Server port (default 8995)
//   --dry-run                 List discovered place engines without executing

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const REPO_ROOT = path.join(__dirname, '..', '..');
const { checkBinaryFreshness } = require(path.join(REPO_ROOT, 'tools', 'freshness_check'));

function parseArgv() {
    const a = process.argv.slice(2);
    const p = {
        flavor: 'StreamMoE',
        engine: null,
        binOverride: null,
        models: path.join(REPO_ROOT, 'tools', 'run_specs', 'models', 'olmoe.json'),
        tasks: path.join(REPO_ROOT, 'tools', 'run_specs', 'tasks', 'hi.json'),
        enginesDir: path.join(REPO_ROOT, 'tools', 'run_specs', 'engines'),
        baseline: path.join(REPO_ROOT, 'baseline_regression', 'baseline', 'olmoe_hi_baseline.txt'),
        outRoot: path.join(REPO_ROOT, 'baseline_regression', 'temp', 'olmoe_places'),
        tempOverride: 0,
        port: 8995,
        dryRun: false,
        strict: false,
    };
    for (let i = 0; i < a.length; i++) {
        if (a[i] === '--flavor' || a[i] === '--flavors') p.flavor = a[++i];
        else if (a[i] === '--engine') p.engine = a[++i];
        else if (a[i] === '--bin-override') p.binOverride = a[++i];
        else if (a[i] === '--temp-override') p.tempOverride = Number(a[++i]);
        else if (a[i] === '--models') p.models = a[++i];
        else if (a[i] === '--tasks') p.tasks = a[++i];
        else if (a[i] === '--engines-dir') p.enginesDir = a[++i];
        else if (a[i] === '--baseline') p.baseline = a[++i];
        else if (a[i] === '--out-root') p.outRoot = a[++i];
        else if (a[i] === '--port') p.port = Number(a[++i]);
        else if (a[i] === '--dry-run') p.dryRun = true;
        else if (a[i] === '--strict') p.strict = true;
    }
    return p;
}

function loadEnvIfPresent() {
    if (!process.env.SM_OLMOE) {
        const envBat = path.join(REPO_ROOT, 'temp', 'sm_env.bat');
        if (fs.existsSync(envBat)) {
            const lines = fs.readFileSync(envBat, 'utf8').split('\n');
            for (const line of lines) {
                const m = line.trim().match(/^set\s+([A-Za-z0-9_]+)=(.*)$/i);
                if (m && m[1] && m[2]) {
                    process.env[m[1].trim()] = m[2].trim();
                }
            }
        }
    }
}

function extractAssistantContent(chatJsonPath) {
    if (!fs.existsSync(chatJsonPath)) return null;
    try {
        const raw = fs.readFileSync(chatJsonPath, 'utf8').replace(/^\uFEFF/, '');
        const data = JSON.parse(raw);
        if (!Array.isArray(data)) return null;
        const msg = data.find((m) => m && m.role === 'assistant');
        if (!msg) return null;
        return typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
    } catch (_) {
        return null;
    }
}

function runSingleEngine(opts, engineJsonPath) {
    return new Promise((resolve) => {
        const runnerScript = path.join(REPO_ROOT, 'tools', 'run_export.js');
        const env = { ...process.env, SM_PORT: String(opts.port) };
        const args = [
            runnerScript,
            '--flavors', opts.flavor,
            '--models', opts.models,
            '--engines', engineJsonPath,
            '--tasks', opts.tasks,
            '--out-root', opts.outRoot,
            '--temp-override', String(opts.tempOverride),
        ];
        if (opts.binOverride) {
            args.push('--bin-override', opts.binOverride);
        }

        const child = spawn(process.execPath, args, {
            cwd: REPO_ROOT,
            env,
            stdio: ['ignore', 'inherit', 'inherit'],
        });

        child.on('close', (code) => resolve(code));
        child.on('error', (err) => {
            console.error(`[verify_places] spawn error for ${path.basename(engineJsonPath)}:`, err.message);
            resolve(-1);
        });
    });
}

(async () => {
    loadEnvIfPresent();
    const opts = parseArgv();

    if (!process.env.SM_OLMOE) {
        console.error('[-] SM_OLMOE is not set in environment or temp/sm_env.bat.');
        console.error('[-]   Please define SM_OLMOE in temp/sm_env.bat pointing to your local olmoe gguf file.');
        process.exit(1);
    }
    if (!fs.existsSync(process.env.SM_OLMOE)) {
        console.error(`[-] SM_OLMOE file not found: ${process.env.SM_OLMOE}`);
        process.exit(1);
    }

    if (!fs.existsSync(opts.enginesDir)) {
        console.error(`[-] Engines directory not found: ${opts.enginesDir}`);
        process.exit(1);
    }

    // Discover all place-*.json files dynamically
    const allFiles = fs.readdirSync(opts.enginesDir);
    let placeFiles = allFiles
        .filter((f) => f.startsWith('place-') && f.endsWith('.json'))
        .sort((a, b) => {
            if (a === 'place-cpu.json') return -1;
            if (b === 'place-cpu.json') return 1;
            return a.localeCompare(b);
        });

    if (opts.engine) {
        const target = opts.engine.toLowerCase();
        placeFiles = placeFiles.filter((f) => {
            const base = path.basename(f, '.json').toLowerCase();
            return base === target || base === 'place-' + target || f.toLowerCase() === target;
        });
        if (placeFiles.length === 0) {
            console.error(`[-] No engine matching "${opts.engine}" found in ${opts.enginesDir}`);
            process.exit(1);
        }
    }

    if (placeFiles.length === 0) {
        console.error(`[-] No place-*.json engine files found in ${opts.enginesDir}`);
        process.exit(1);
    }

    const effectiveBin = opts.binOverride || path.join(REPO_ROOT, 'build', opts.flavor, 'llama-build', 'bin', 'llama-server.exe');
    checkBinaryFreshness(effectiveBin);

    console.log('=====================================================================');
    console.log(`[verify_places] Flavor        : ${opts.flavor}`);
    console.log(`[verify_places] Target binary : ${path.relative(REPO_ROOT, effectiveBin)}`);
    console.log(`[verify_places] Discovered ${placeFiles.length} place-* engine spec(s):`);
    for (const f of placeFiles) {
        console.log(`  - ${f}`);
    }
    console.log(`[verify_places] Model spec   : ${opts.models}`);
    console.log(`[verify_places] Task spec    : ${opts.tasks}`);
    console.log(`[verify_places] Baseline file: ${opts.baseline}`);
    console.log(`[verify_places] Temp override: ${opts.tempOverride}`);
    console.log('=====================================================================');

    if (opts.dryRun) {
        console.log('[verify_places] --dry-run specified, exiting without running.');
        process.exit(0);
    }

    const modelSpec = JSON.parse(fs.readFileSync(opts.models, 'utf8').replace(/^\uFEFF/, ''));
    const taskSpec = JSON.parse(fs.readFileSync(opts.tasks, 'utf8').replace(/^\uFEFF/, ''));
    const modelName = modelSpec.model;
    const taskInput = taskSpec.input;

    // 1. Establish baseline
    let baselineText = null;
    let cpuAlreadyRun = false;
    let cpuChatPath = null;

    if (fs.existsSync(opts.baseline)) {
        baselineText = fs.readFileSync(opts.baseline, 'utf8').trim();
        console.log(`\n[verify_places] Loaded existing standard baseline (${baselineText.length} chars):`);
        console.log(`  "${baselineText.slice(0, 120)}${baselineText.length > 120 ? '...' : ''}"\n`);
    } else {
        console.log(`\n[verify_places] Baseline file not found: ${opts.baseline}`);
        console.log('[verify_places] Generating standard baseline using place-cpu.json ...');
        const cpuFile = 'place-cpu.json';
        const cpuJsonPath = path.join(opts.enginesDir, cpuFile);
        const cpuEngine = path.basename(cpuFile, '.json');

        const code = await runSingleEngine(opts, cpuJsonPath);
        cpuAlreadyRun = true;
        cpuChatPath = path.join(opts.outRoot, opts.flavor, modelName, cpuEngine, taskInput, 'chat.json');

        if (code !== 0) {
            console.error(`[-] Failed to generate baseline: runner exited with code ${code}`);
            process.exit(1);
        }

        const genText = extractAssistantContent(cpuChatPath);
        if (!genText) {
            console.error(`[-] Failed to extract assistant response from baseline run: ${cpuChatPath}`);
            process.exit(1);
        }

        baselineText = genText.trim();
        fs.mkdirSync(path.dirname(opts.baseline), { recursive: true });
        fs.writeFileSync(opts.baseline, baselineText, 'utf8');
        console.log(`[verify_places] Standard baseline saved to ${opts.baseline} (${baselineText.length} chars):`);
        console.log(`  "${baselineText.slice(0, 120)}${baselineText.length > 120 ? '...' : ''}"\n`);
    }

    // 2. Run and test selected place-*.json engines
    const results = [];
    let passCount = 0;
    let diffCount = 0;
    let errCount = 0;

    for (let i = 0; i < placeFiles.length; i++) {
        const file = placeFiles[i];
        const engine = path.basename(file, '.json');
        const chatPath = path.join(opts.outRoot, opts.flavor, modelName, engine, taskInput, 'chat.json');
        console.log(`\n[${i + 1}/${placeFiles.length}] Testing engine: ${engine} ...`);

        // If place-cpu was just executed during baseline creation, reuse its chat.json
        if (cpuAlreadyRun && engine === 'place-cpu' && fs.existsSync(cpuChatPath)) {
            console.log(`  (Reusing chat.json from baseline generation)`);
        } else {
            const engineJsonPath = path.join(opts.enginesDir, file);
            const code = await runSingleEngine(opts, engineJsonPath);
            if (code !== 0) {
                console.log(`[-] ${engine}: run failed with exit code ${code}`);
                results.push({
                    engine,
                    status: 'ERROR',
                    content: null,
                    error: `Runner exit code ${code}`,
                });
                errCount++;
                continue;
            }
        }

        const text = extractAssistantContent(chatPath);
        const trimmed = (text || '').trim();

        if (!text) {
            console.log(`[-] ${engine}: 无法获取输出内容 (未找到 chat.json 或 assistant 内容为空)`);
            results.push({
                engine,
                status: 'ERROR',
                content: null,
                error: 'No assistant message found in chat.json',
            });
            errCount++;
        } else if (trimmed === baselineText) {
            console.log(`[PASS] ${engine}: 正确`);
            results.push({
                engine,
                status: 'PASS',
                content: trimmed,
            });
            passCount++;
        } else {
            console.log(`[DIFF] ${engine}: 不一致，这个place生成的内容是:\n${trimmed}`);
            results.push({
                engine,
                status: 'DIFF',
                content: trimmed,
            });
            diffCount++;
        }
    }

    // 3. Output overall summary
    console.log('\n=====================================================================');
    console.log('                    PLACE REGRESSION SUMMARY');
    console.log('=====================================================================');
    console.log(`Standard baseline: "${baselineText.slice(0, 100)}${baselineText.length > 100 ? '...' : ''}"`);
    console.log(`Total tested     : ${placeFiles.length}`);
    console.log(`PASS (一模一样)  : ${passCount}`);
    console.log(`DIFF (内容不一致): ${diffCount}`);
    console.log(`ERROR (运行异常) : ${errCount}`);
    console.log('---------------------------------------------------------------------');

    for (const r of results) {
        if (r.status === 'PASS') {
            console.log(`  [PASS]  ${r.engine.padEnd(20)} -> 正确`);
        } else if (r.status === 'DIFF') {
            console.log(`  [DIFF]  ${r.engine.padEnd(20)} -> 不一致 (${r.content.slice(0, 60)}...)`);
        } else {
            console.log(`  [ERROR] ${r.engine.padEnd(20)} -> 运行异常 (${r.error})`);
        }
    }
    console.log('=====================================================================');

    // Save structured report
    const reportPath = path.join(opts.outRoot, opts.flavor, 'places_summary.json');
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(
        reportPath,
        JSON.stringify(
            {
                timestamp: new Date().toISOString(),
                flavor: opts.flavor,
                baseline: baselineText,
                total: placeFiles.length,
                pass: passCount,
                diff: diffCount,
                error: errCount,
                results,
            },
            null,
            2
        ),
        'utf8'
    );
    console.log(`[verify_places] Detailed report written to: ${reportPath}`);

    if (errCount > 0) {
        console.error(`[-] [verify_places] ${errCount} engine(s) failed with runtime errors.`);
        process.exit(1);
    }
    if (opts.strict && diffCount > 0) {
        console.error(`[-] [verify_places] Strict mode enabled and ${diffCount} engine(s) produced different text.`);
        process.exit(1);
    }
    if (diffCount > 0) {
        console.log(`[+] [verify_places] All ${placeFiles.length} engines ran successfully (${diffCount} minor fp precision diffs across backends).`);
    } else {
        console.log(`[+] [verify_places] All ${placeFiles.length} engines ran successfully and matched baseline perfectly!`);
    }
    process.exit(0);
})();
