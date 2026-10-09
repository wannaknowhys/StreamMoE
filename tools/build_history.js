// tools/build_history.js
// History-binary builder (driven ONLY via `build.bat history ...`).
//
// Flow: check out <commit> in a disposable worktree at temp/wt_<short>/,
// update the recorded submodule, apply that commit's own vendored patch stack,
// then build plain `llamalibs <tag>` with THAT commit's own build.bat recipe
// (old recipes only know plain tags; <tag>_<hash> would silently match no
// feature mapping). Finally move only bin/ (exe+dll+pdb) to the main repo at
// build/<tag>_<short>/, write a manifest, and remove the worktree.
//
// Failure policy: any failure keeps the worktree as the scene and aborts with
// the resume path printed. Success removes the worktree. History outputs are
// one-shot snapshots: the moved dir is NOT incrementally rebuildable (cmake
// cache holds worktree-absolute paths), recorded in the manifest.
'use strict';
const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const MAIN_ROOT = path.resolve(__dirname, '..');
const TEMP_DIR = path.join(MAIN_ROOT, 'temp');
const BUILD_DIR = path.join(MAIN_ROOT, 'build');
// Canonical vendored patch stack (oldest-first). Only entries present in the
// historical commit are applied; anything else under patches/ is opt-in
// diagnostics (e.g. memwatch) and is reported but never auto-applied.
// (2026-10-09: prefill-export-llama.patch eliminated - prefill export is now
// phase-1 anchors + frags, so the stack is macros -> tsc_timer -> route-b.)
const PATCH_STACK = [
  'patches/streammoe-macros.patch',
  'patches/tsc_timer.patch',
  'patches/route-b-inject.patch',
];

function info(m) { console.log('[history] ' + m); }
function fail(m) { console.error('[-] history: ' + m); process.exit(1); }

function git(args, cwd) {
  return spawnSync('git', args, { cwd: cwd || MAIN_ROOT, encoding: 'utf8' });
}

// Streaming run with a tee to logFile (long ninja builds stay watchable).
function runTee(cmd, args, cwd, logFile) {
  return new Promise((resolve) => {
    const log = fs.createWriteStream(logFile, { encoding: 'utf8' });
    let settled = false;
    const done = (code, err) => {
      if (settled) return;
      settled = true;
      log.end(() => resolve({ code, err }));
    };
    let p;
    try {
      p = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (e) { done(1, e); return; }
    p.stdout.on('data', (d) => { process.stdout.write(d); log.write(d); });
    p.stderr.on('data', (d) => { process.stderr.write(d); log.write(d); });
    p.on('error', (e) => done(1, e));
    p.on('close', (code) => done(code == null ? 1 : code, null));
  });
}

function moveDir(src, dst) {
  try {
    fs.renameSync(src, dst);
    return;
  } catch (e) {
    if (e.code !== 'EXDEV') throw e;
  }
  fs.cpSync(src, dst, { recursive: true });
  fs.rmSync(src, { recursive: true, force: true });
}

function parseWorktrees() {
  const r = git(['worktree', 'list', '--porcelain']);
  if (r.status !== 0) fail('git worktree list failed:\n' + r.stderr);
  const out = [];
  let cur = null;
  for (const line of r.stdout.split('\n')) {
    if (line.startsWith('worktree ')) { cur = { path: line.slice(9).trim(), head: '' }; out.push(cur); }
    else if (line.startsWith('HEAD ') && cur) cur.head = line.slice(5).trim();
  }
  return out;
}

function usage() {
  console.error('usage: build.bat history <commit> <tag...> [--force] [--allow-unknown]');
  console.error('       build.bat history-clean <commit>');
  console.error('  tags are space-separated (a quoted "tag1,tag2" also works).');
  process.exit(1);
}

async function main() {
  let argv = process.argv.slice(2);
  // Tolerate pass-through of the bat subcommand word (`node script %*` style).
  if (argv[0] === 'history' || argv[0] === 'history-clean') {
    if (argv[0] === 'history-clean') argv = ['--clean'].concat(argv.slice(1));
    else argv = argv.slice(1);
  }
  const flags = new Set(argv.filter((a) => a.startsWith('--')));
  const cleanMode = flags.has('--clean');
  const force = flags.has('--force');
  const allowUnknown = flags.has('--allow-unknown');
  const positional = argv.filter((a) => !a.startsWith('--'));
  const commitIn = positional[0];
  const tags = positional.slice(1).join(',').split(',').map((s) => s.trim()).filter(Boolean);
  if (!commitIn || (tags.length === 0 && !cleanMode)) usage();

  const rv = git(['rev-parse', '--verify', commitIn + '^{commit}']);
  if (rv.status !== 0) fail('unknown commit: ' + commitIn);
  const full = rv.stdout.trim();
  const short = git(['rev-parse', '--short', full]).stdout.trim();
  const wtPath = path.join(TEMP_DIR, 'wt_' + short);

  if (cleanMode) {
    const r = git(['worktree', 'remove', '--force', '--', wtPath]);
    git(['worktree', 'prune']);
    if (r.status !== 0) fail('worktree remove failed:\n' + r.stderr);
    info('removed ' + wtPath);
    return;
  }

  fs.mkdirSync(TEMP_DIR, { recursive: true });
  const wts = parseWorktrees();
  const samePath = wts.find((w) => path.resolve(w.path).toLowerCase() === path.resolve(wtPath).toLowerCase());
  let reuse = false;
  if (samePath) {
    if (samePath.head.toLowerCase() === full.toLowerCase()) {
      reuse = true;
      info('reusing worktree ' + wtPath);
    } else if (!force) {
      fail(wtPath + ' is registered for another commit.\n' +
        'Refusing to delete it. Rerun with --force, or: build.bat history-clean ' + samePath.head.slice(0, 12));
    } else {
      info('--force: recreating worktree for ' + short);
      if (git(['worktree', 'remove', '--force', '--', wtPath]).status !== 0) fail('worktree remove failed');
    }
  } else if (fs.existsSync(wtPath) && !force) {
    fail(wtPath + ' exists but is not a registered worktree. Remove it manually or rerun with --force.');
  } else if (fs.existsSync(wtPath) && force) {
    fs.rmSync(wtPath, { recursive: true, force: true });
  }
  git(['worktree', 'prune']);

  if (!reuse) {
    info('adding worktree ' + wtPath + ' @ ' + short);
    const r = git(['worktree', 'add', '--detach', '--', wtPath, full]);
    if (r.status !== 0) fail('worktree add failed:\n' + r.stderr);
  }

  // Submodule pinned by the historical commit (faithful to that time).
  {
    const r = git(['-C', wtPath, 'submodule', 'update', '--init', '--', 'third_party/llama.cpp']);
    if (r.status !== 0) fail('submodule update failed, scene kept at ' + wtPath + ':\n' + r.stderr);
  }
  const sub = path.join(wtPath, 'third_party', 'llama.cpp');

  // Tags must be known to the HISTORICAL build.bat, else its feature mapping
  // silently builds the wrong flavor (empty STREAM_MOE_FEATURES).
  {
    let bat = '';
    try { bat = fs.readFileSync(path.join(wtPath, 'build.bat'), 'utf8'); } catch (e) { fail('cannot read historical build.bat: ' + e.message); }
    for (const t of tags) {
      if (!bat.includes('"%TAG%"=="' + t + '"') && !allowUnknown) {
        fail('tag "' + t + '" is unknown to ' + short + ':build.bat (would silently build the wrong flavor).\n' +
          'Scene kept at ' + wtPath + '. Rerun with --allow-unknown to override.');
      }
    }
  }

  // Apply that commit's own vendored patch stack (all of it: the historical
  // worktree state is the full stack applied, never a subset).
  // Stacked patches overlap in the same files, so once the full stack is on,
  // per-patch forward/reverse --check both fail for every patch below the top.
  // Only the TOP patch reverse-checks reliably (nothing sits above it), plus a
  // marker file records exactly which patch bytes were applied.
  const crypto = require('node:crypto');
  const sha256 = (p) => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  const present = PATCH_STACK.filter((rel) => fs.existsSync(path.join(wtPath, rel)));
  const shas = {};
  for (const rel of present) shas[rel] = sha256(path.join(wtPath, rel));
  const markerPath = path.join(wtPath, '.history_stack.json');
  let stackOk = false;
  const applied = [];
  try {
    const m = JSON.parse(fs.readFileSync(markerPath, 'utf8'));
    if (m.commit === full && JSON.stringify(m.patches) === JSON.stringify(shas) &&
        git(['-C', sub, 'status', '--porcelain']).stdout.trim() !== '') {
      stackOk = true;
      for (const rel of present) applied.push(rel + ' (marker)');
      info('patch stack verified by marker, skip');
    }
  } catch (e) { /* no usable marker */ }
  if (!stackOk && present.length > 0) {
    const top = present[present.length - 1];
    if (git(['-C', sub, 'apply', '--reverse', '--check', '--', path.join(wtPath, top)]).status === 0) {
      stackOk = true;
      for (const rel of present) applied.push(rel + ' (assumed: top of stack present)');
      info('top of stack already applied, assuming full stack, skip');
    }
  }
  if (!stackOk) {
    for (const rel of present) {
      const p = path.join(wtPath, rel);
      if (git(['-C', sub, 'apply', '--check', '--', p]).status === 0) {
        const r = git(['-C', sub, 'apply', '--', p]);
        if (r.status !== 0) fail('apply failed for ' + rel + ', scene kept at ' + wtPath + ':\n' + r.stderr);
        applied.push(rel);
        info('applied ' + rel);
      } else if (git(['-C', sub, 'apply', '--reverse', '--check', '--', p]).status === 0) {
        applied.push(rel + ' (already)');
        info('already applied, skip ' + rel);
      } else {
        fail('patch does not apply: ' + rel + ', scene kept at ' + wtPath);
      }
    }
  }
  try {
    fs.writeFileSync(markerPath, JSON.stringify({ commit: full, patches: shas }, null, 2), 'utf8');
  } catch (e) { info('cannot write stack marker (non-fatal): ' + e.message); }
  try {
    const others = fs.readdirSync(path.join(wtPath, 'patches'))
      .filter((f) => f.endsWith('.patch'))
      .filter((f) => !PATCH_STACK.some((s) => s.endsWith('/' + f)));
    if (others.length > 0) info('not in stack, left unapplied: ' + others.join(', '));
  } catch (e) { /* historical tree without patches/, nothing to report */ }

  for (const tag of tags) {
    const dest = path.join(BUILD_DIR, tag + '_' + short);
    const destExe = path.join(dest, 'bin', 'llama-server.exe');
    // llamalibs binaries live in <tag>/llama-build/bin (ninja -C LLAMA_BUILD);
    // tag-root bin/ is only for test/convert/harness outputs.
    const wtBin = path.join(wtPath, 'build', tag, 'llama-build', 'bin');
    const wtExe = path.join(wtBin, 'llama-server.exe');
    if (fs.existsSync(destExe) && !force) {
      info('already done, skip ' + tag + ' -> ' + dest);
      continue;
    }
    const logFile = path.join(TEMP_DIR, 'history_' + short + '_' + tag + '.log');
    if (fs.existsSync(wtExe)) {
      info('worktree binary found, move without rebuild: ' + tag);
    } else {
      info('building llamalibs ' + tag + ' @ ' + short + ' (log ' + logFile + ')');
      const res = await runTee('cmd.exe', ['/c', 'build.bat', 'llamalibs', tag], wtPath, logFile);
      if (res.code !== 0 || res.err) {
        fail('build failed for tag ' + tag + ' (code ' + res.code + '). Scene kept at ' + wtPath + ', log ' + logFile);
      }
      if (!fs.existsSync(wtExe)) {
        fail('expected binary missing: ' + wtExe + '. Scene kept at ' + wtPath + ', log ' + logFile);
      }
    }
    if (fs.existsSync(dest)) fs.rmSync(dest, { recursive: true, force: true });
    fs.mkdirSync(dest, { recursive: true });
    moveDir(wtBin, path.join(dest, 'bin'));
    if (fs.existsSync(logFile)) {
      try { fs.renameSync(logFile, path.join(dest, 'build.log')); } catch (e) { info('log left at ' + logFile); }
    }
    const manifest = {
      commit: full,
      short: short,
      tag: tag,
      built_at_utc: new Date().toISOString(),
      builder: 'tools/build_history.js via build.bat history',
      patches_applied: applied,
      note: 'One-shot snapshot: cmake intermediates were discarded; this dir is NOT incrementally rebuildable.',
    };
    fs.writeFileSync(path.join(dest, 'history_manifest.json'), JSON.stringify(manifest, null, 2), 'utf8');
    info('done -> ' + dest);
  }

  const r = git(['worktree', 'remove', '--force', '--', wtPath]);
  git(['worktree', 'prune']);
  if (r.status !== 0) {
    console.error('[-] history: artifacts moved, but worktree remove failed. Clean manually: git worktree remove --force ' + wtPath);
    process.exit(1);
  }
  info('worktree removed, all tags done.');
}

main().catch((e) => fail('unexpected: ' + (e && e.stack || e)));
