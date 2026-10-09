// tools/freshness_check.js
// Guardrail: checks that target binaries are compiled AFTER the latest source/patch modification.
// Usage as module:
//   const { checkBinaryFreshness } = require('./freshness_check');
//   checkBinaryFreshness(binPath);
// Usage as CLI:
//   node tools/freshness_check.js <binPath> [<binPath2> ...] [--no-fresh-check]

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..');

const DEFAULT_SOURCE_ROOTS = [
    path.join(REPO_ROOT, 'src'),
    path.join(REPO_ROOT, 'patches'),
];

const TRACKED_EXTENSIONS = new Set(['.cpp', '.c', '.h', '.hpp', '.frag', '.patch']);

function getLatestSourceMtime(roots = DEFAULT_SOURCE_ROOTS) {
    let latest = { time: 0, file: '' };

    function walk(dir) {
        if (!fs.existsSync(dir)) return;
        let entries = [];
        try {
            entries = fs.readdirSync(dir, { withFileTypes: true });
        } catch (_) {
            return;
        }

        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                if (entry.name === '.git' || entry.name === 'temp' || entry.name === 'build') continue;
                walk(fullPath);
            } else if (entry.isFile()) {
                const ext = path.extname(entry.name).toLowerCase();
                if (TRACKED_EXTENSIONS.has(ext)) {
                    try {
                        const st = fs.statSync(fullPath);
                        if (st.mtimeMs > latest.time) {
                            latest = { time: st.mtimeMs, file: fullPath };
                        }
                    } catch (_) {}
                }
            }
        }
    }

    for (const r of roots) {
        walk(r);
    }

    // Also inspect dirty/modified files in third_party/llama.cpp
    try {
        const { execSync } = require('child_process');
        const out = execSync('git -C third_party/llama.cpp status --porcelain', { cwd: REPO_ROOT, encoding: 'utf8', timeout: 2000, stdio: ['ignore', 'pipe', 'ignore'] });
        const lines = out.split('\n');
        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            const rel = trimmed.slice(2).trim();
            const fullPath = path.join(REPO_ROOT, 'third_party', 'llama.cpp', rel);
            if (fs.existsSync(fullPath)) {
                try {
                    const st = fs.statSync(fullPath);
                    if (st.mtimeMs > latest.time) {
                        latest = { time: st.mtimeMs, file: fullPath };
                    }
                } catch (_) {}
            }
        }
    } catch (_) {}

    return latest;
}

function checkBinaryFreshness(binPath, options = {}) {
    const bypass = options.bypass || process.argv.includes('--no-fresh-check') || process.env.SM_NO_FRESH_CHECK === '1';
    if (bypass) {
        if (options.verbose) {
            console.log('[freshness_check] WARNING: Freshness check bypassed (--no-fresh-check)');
        }
        return true;
    }

    const resolvedBin = path.isAbsolute(binPath) ? binPath : path.resolve(REPO_ROOT, binPath);
    if (!fs.existsSync(resolvedBin)) {
        const msg = `[freshness_check] ERROR: Target binary not found: ${resolvedBin}`;
        if (options.strict !== false) {
            console.error(`\n[-] =====================================================================`);
            console.error(`[-] ${msg}`);
            console.error(`[-] Please run: .\\build.bat llamalibs dual`);
            console.error(`[-] =====================================================================\n`);
            process.exit(1);
        }
        return false;
    }

    const binStat = fs.statSync(resolvedBin);
    const binMtime = binStat.mtimeMs;
    const latestSrc = getLatestSourceMtime(options.roots || DEFAULT_SOURCE_ROOTS);

    // If source file is newer than binary by more than 1000ms (1s clock skew tolerance)
    if (latestSrc.time > 0 && binMtime + 1000 < latestSrc.time) {
        const lagSec = ((latestSrc.time - binMtime) / 1000).toFixed(1);
        const relSrc = path.relative(REPO_ROOT, latestSrc.file);
        const relBin = path.relative(REPO_ROOT, resolvedBin);

        if (options.strict !== false) {
            console.error(`\n[-] =====================================================================`);
            console.error(`[-] [freshness_check] ERROR: Target binary is STALE!`);
            console.error(`[-]   Binary : ${relBin}`);
            console.error(`[-]            Built on: ${new Date(binMtime).toLocaleString()}`);
            console.error(`[-]   Source : ${relSrc}`);
            console.error(`[-]            Modified: ${new Date(latestSrc.time).toLocaleString()} (${lagSec}s AFTER binary)`);
            console.error(`[-]`);
            console.error(`[-] A source code change has NOT been compiled into this binary!`);
            console.error(`[-] Please recompile both release & test binaries before running:`);
            console.error(`[-]   .\\build.bat llamalibs dual`);
            console.error(`[-] (or pass --no-fresh-check to bypass this guard)`);
            console.error(`[-] =====================================================================\n`);
            process.exit(1);
        }
        return false;
    }

    if (options.verbose) {
        console.log(`[freshness_check] OK: ${path.basename(resolvedBin)} is fresh (vs ${path.relative(REPO_ROOT, latestSrc.file)})`);
    }
    return true;
}

if (require.main === module) {
    const args = process.argv.slice(2).filter((a) => !a.startsWith('--'));
    if (args.length === 0) {
        console.log('Usage: node tools/freshness_check.js <binPath> [<binPath2> ...] [--no-fresh-check]');
        process.exit(0);
    }
    for (const b of args) {
        checkBinaryFreshness(b, { verbose: true, strict: true });
    }
}

module.exports = {
    checkBinaryFreshness,
    getLatestSourceMtime,
};
