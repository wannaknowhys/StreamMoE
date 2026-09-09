// check_tensor_data.js - iron-rule guard (docs/GRAPH_PARTITION.md):
// forbid raw memcpy/memmove/memset on ggml_tensor::data outside a backend's
// own iface implementation. Cross-platform, zero dependencies.
//
//   node scripts/check_tensor_data.js
//
// Exit 0 = clean, 1 = violation(s). Exempt a line (or the line after it) with
// the marker comment `iron-rule-exempt` for legitimate raw-arena / iface copies.
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const DIRS = ['src', 'patches'];
const EXT = new Set(['.cpp', '.h', '.frag']);
// memcpy/memmove/memset whose argument list touches tensor ->data.
const RE = /\b(?:std::)?(memcpy|memmove|memset)\s*\([^;]*->data\b/;
const EXEMPT = /iron-rule-exempt/;

function* walk(dir) {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const p = path.join(dir, e.name);
        if (e.isDirectory()) yield* walk(p);
        else if (EXT.has(path.extname(e.name))) yield p;
    }
}

let violations = 0, scanned = 0;
for (const dir of DIRS) {
    const abs = path.join(ROOT, dir);
    if (!fs.existsSync(abs)) continue;
    for (const file of walk(abs)) {
        scanned++;
        const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/);
        for (let i = 0; i < lines.length; i++) {
            if (!RE.test(lines[i])) continue;
            if (EXEMPT.test(lines[i]) || (i > 0 && EXEMPT.test(lines[i - 1]))) continue;
            const rel = path.relative(ROOT, file).replace(/\\/g, '/');
            console.log(`${rel}:${i + 1}: ${lines[i].trim()}`);
            violations++;
        }
    }
}

console.log(`\n[check_tensor_data] scanned ${scanned} files, ${violations} raw tensor-data copy site(s)`);
if (violations) {
    console.log('[check_tensor_data] FAIL - use ggml_backend_tensor_get/set (backend/tensor_io.h) instead.');
    console.log('                   See docs/GRAPH_PARTITION.md iron rule.');
    process.exit(1);
}
console.log('[check_tensor_data] PASS');
