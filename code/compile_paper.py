"""Compile paper/main.tex and then read the produced PDF back, looking for the failures that a
successful compile does not catch.

"It compiled" proves almost nothing. A bare `%` injected from Python turns the rest of its line
into a comment, and the truncated sentence usually still reads fluently; an undefined reference
renders as `??`; a missing value renders as the literal `nan`. All of those survive a clean exit
code, so the PDF's extracted text is scanned for each of them and a hit is a hard failure.

Usage: python code/compile_paper.py
"""
import os
import re
import subprocess
import sys

BASE = "/home/snakehand/Documents/mdpi_papers/AIIA"
TEC = "/home/snakehand/micromamba/envs/tectonic/bin/tectonic"

CHECKS = [
    ("unresolved cross-reference", re.compile(r"\?\?")),
    ("literal nan", re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])")),
    ("TODO marker", re.compile(r"TODO|FIXME|XXX")),
    ("bare LaTeX macro", re.compile(r"\\[A-Za-z]{3,}")),
    ("empty braces", re.compile(r"\{\s*\}")),
    ("python repr leaked", re.compile(r"(\[\s*'|\{'|dtype=|Index\(|array\()")),
    # a digit immediately followed by a letter is the signature of a swallowed percent sign:
    # "41.9% : screening" compiles to "41.9with ..." when the % starts a comment
    ("digit followed by letter (swallowed %?)", re.compile(r"\d(?:with|and|of|the|in|on|for|to|was|were|is)\b")),
]
ALLOW = re.compile(r"\d+\s*(dB|ms|MB|GB|M|G|px)\b")


def main():
    tex = f"{BASE}/paper/main.tex"
    if not os.path.exists(tex):
        sys.exit("paper/main.tex not found -- run code/write_paper.py first")
    r = subprocess.run([TEC, "-X", "compile", "--keep-logs", "--outdir", f"{BASE}/paper", tex],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-4000:])
        sys.exit(f"tectonic failed (rc={r.returncode})")
    pdf = f"{BASE}/paper/main.pdf"
    import pymupdf
    with pymupdf.open(pdf) as d:
        pages = d.page_count
        text = "\n".join(p.get_text() for p in d)
    open(f"{BASE}/paper/main.txt", "w").write(text)

    bad = []
    for name, pat in CHECKS:
        hits = [m.group(0) for m in pat.finditer(text)]
        hits = [h for h in hits if not ALLOW.search(h)]
        if hits:
            bad.append((name, len(hits), sorted(set(hits))[:6]))
    nfig = len(re.findall(r"\n\s*Figure\s+\d+", text))
    ntab = len(re.findall(r"\n\s*Table\s+\d+", text))
    print(f"compiled: {pages} pages, {nfig} figure captions, {ntab} table captions")
    for name, n, ex in bad:
        print(f"  FAIL {name}: {n} hit(s) {ex}")
    if bad:
        return 1
    print("PDF text checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
