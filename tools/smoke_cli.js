// tools/smoke_cli.js
// Ultra-fast CLI smoke test: ensures llama-cli with mixed pools generates coherent text.
// Usage:
//   node tools/smoke_cli.js [--bin <path>] [--model <path>] [--no-fresh-check]

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { checkBinaryFreshness } = require('./freshness_check');

const REPO_ROOT = path.resolve(__dirname, '..');

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

function parseArgs() {
    const args = process.argv.slice(2);
    const opts = {
        bin: path.join(REPO_ROOT, 'build', 'StreamMoE', 'llama-build', 'bin', 'llama-cli.exe'),
        model: null,
        nTokens: 16,
    };
    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--bin') opts.bin = path.resolve(args[++i]);
        else if (args[i] === '--model') opts.model = path.resolve(args[++i]);
        else if (args[i] === '-n') opts.nTokens = Number(args[++i]);
    }
    return opts;
}

(function main() {
    loadEnvIfPresent();
    const opts = parseArgs();

    const modelPath = opts.model || process.env.SM_OLMOE;
    if (!modelPath || !fs.existsSync(modelPath)) {
        console.error(`[-] [smoke_cli] ERROR: Model file not found: ${modelPath}`);
        console.error(`[-]   Please set SM_OLMOE in temp/sm_env.bat or pass --model <path>`);
        process.exit(1);
    }

    // 1. Freshness check
    checkBinaryFreshness(opts.bin);

    console.log(`[smoke_cli] Running llama-cli smoke test...`);
    console.log(`  Binary: ${path.relative(REPO_ROOT, opts.bin)}`);
    console.log(`  Model : ${path.basename(modelPath)}`);

    const t0 = Date.now();
    const child = spawnSync(
        opts.bin,
        [
            '--expert-backend',
            '--moe-expert-pools', 'RAM:8192,Vulkan0:5120',
            '-p', 'hi',
            '-st',
            '-m', modelPath,
            '-n', String(opts.nTokens),
            '--temp', '0',
        ],
        {
            input: '\n',
            encoding: 'utf8',
            timeout: 60000,
        }
    );

    const elapsed = ((Date.now() - t0) / 1000).toFixed(2);

    if (child.error) {
        console.error(`[-] [smoke_cli] ERROR: Execution failed: ${child.error.message}`);
        process.exit(1);
    }
    if (child.status !== 0) {
        console.error(`[-] [smoke_cli] ERROR: Exited with code ${child.status}`);
        console.error(child.stderr || child.stdout);
        process.exit(child.status || 1);
    }

    const output = child.stdout || '';

    // Extract text after prompt '> hi'
    const marker = '> hi';
    const markerIdx = output.indexOf(marker);
    let genText = '';
    if (markerIdx !== -1) {
        genText = output.slice(markerIdx + marker.length);
        const promptIdx = genText.indexOf('[');
        if (promptIdx !== -1) {
            genText = genText.slice(0, promptIdx);
        }
    } else {
        genText = output;
    }
    genText = genText.trim();

    // Gibberish / repetition detection heuristics
    const isGibberish =
        /amongst\s+other/i.test(genText) ||
        /accordion/i.test(genText) ||
        /amongstunto/i.test(genText) ||
        /(.)\1{4,}/.test(genText);

    // Coherence check: should have standard English greeting tokens
    const isCoherent = /\bhello\b|\bhi\b|\bthere\b|how\s+can\s+i|help|assist|today|questions/i.test(genText);

    if (isGibberish || !isCoherent) {
        console.error(`[-] =====================================================================`);
        console.error(`[-] [smoke_cli] FAIL: llama-cli output is GIBBERISH / INCOHERENT!`);
        console.error(`[-]   Time   : ${elapsed}s`);
        console.error(`[-]   Output : "${genText}"`);
        console.error(`[-] =====================================================================`);
        process.exit(1);
    }

    console.log(`[+] [smoke_cli] PASS: llama-cli generated coherent text in ${elapsed}s:`);
    console.log(`    "${genText.slice(0, 100)}${genText.length > 100 ? '...' : ''}"\n`);
    process.exit(0);
})();
