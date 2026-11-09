#!/usr/bin/env python3
"""Compare snapshot directories (per-tensor .bin dumps) or a dir against a
committed .sha256 list.

Usage:
  python tools/compare_snapshots.py DIR_A DIR_B
  python tools/compare_snapshots.py DIR_A --list tools/snapshot_aa12290_ub6.sha256

Prints per-file differences; exits 1 if any file differs or is missing.
Only files present in both sides are compared when DIR_A/DIR_B; the list mode
reports files in the list missing from the dir as well.
"""
import hashlib
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print(__doc__)
        return 2
    a = argv[0]
    if len(argv) == 3 and argv[1] == "--list":
        lst = {}
        with open(argv[2], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                h, name = line.split(None, 1)
                lst[name] = h
        files = sorted(lst)
        diff = 0
        for name in files:
            p = os.path.join(a, name)
            if not os.path.exists(p):
                print("MISSING", name)
                diff += 1
            elif sha256(p) != lst[name]:
                print("DIFF   ", name)
                diff += 1
        extra = [f for f in os.listdir(a) if f not in lst and f.endswith(".bin")]
        for name in extra:
            print("EXTRA  ", name)
        print(f"checked={len(files)} diff={diff} extra={len(extra)}")
        return 0 if diff == 0 else 1
    b = argv[1]
    fa = sorted(f for f in os.listdir(a) if f.endswith(".bin"))
    fb = sorted(f for f in os.listdir(b) if f.endswith(".bin"))
    common = [f for f in fa if f in fb]
    diff = 0
    for name in common:
        if sha256(os.path.join(a, name)) != sha256(os.path.join(b, name)):
            print("DIFF   ", name)
            diff += 1
    only_a = [f for f in fa if f not in fb]
    only_b = [f for f in fb if f not in fa]
    for name in only_a:
        print("ONLY_A ", name)
    for name in only_b:
        print("ONLY_B ", name)
    print(f"a={len(fa)} b={len(fb)} common={len(common)} diff={diff} only_a={len(only_a)} only_b={len(only_b)}")
    return 0 if diff == 0 and not only_a and not only_b else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
