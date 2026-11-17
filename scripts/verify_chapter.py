#!/usr/bin/env python3
"""Check the chapter against the run logs and against its own constraints.

1. Every generated file (numbers, tables, data figures) is regenerated from
   ``results/`` and compared byte for byte with the copy in ``chapter/``.
2. Every number macro defined is used in the manuscript, and every macro used
   is defined.
3. No result is typed by hand: every decimal literal left in the manuscript
   source must be one of the declared design constants below.
4. Publisher constraints: at most three figures, all floats referenced, all
   citations resolvable and all bibliography entries cited, required
   declarations present.

Usage
-----
    python scripts/verify_chapter.py

Exit status 0 when every check passes, 1 otherwise. Run it on the machine that
produced ``results/``: the figure PDFs are compared byte for byte, and fonts and
library versions differ between platforms.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Decimal literals that may appear in the manuscript because they are design
#: constants of the simulator, the algorithm or the statistics -- not results.
DESIGN_CONSTANTS = {
    "0.9", "0.7", "0.1", "0.05", "0.85", "1.15", "1.00", "1.6", "0.995", "0.95",
    "0.003", "0.2", "0.01", "0.5", "1.5", "1.0", "2.0", "0.0", "0.10",
    "4.0",   # the CC BY 4.0 licence version
}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.rows.append((bool(ok), name, detail))

    def summary(self) -> int:
        w = max(len(n) for _, n, _ in self.rows) + 2
        for ok, name, detail in self.rows:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:<{w}} {detail}")
        bad = sum(1 for ok, _, _ in self.rows if not ok)
        print()
        print(f"=== {bad} of {len(self.rows)} CHECKS FAILED ===" if bad
              else f"=== ALL {len(self.rows)} CHECKS PASSED ===")
        return 1 if bad else 0


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chapter", type=Path, default=ROOT / "chapter" / "chapter.tex")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    r = Report()

    # ---- 1. structural constants, from first principles -------------------
    r.check("K = C(k+S-1, S-1) = 15 for k=4, S=3", comb(4 + 3 - 1, 3 - 1) == 15)

    def mlp(i, h, o):
        return i * h + h + h * h + h + h * o + o

    params = mlp(12, 64, 15) + mlp(12, 64, 1)
    r.check("11,024 parameters per agent", params == 11_024, f"computed={params}")
    r.check("observation dimension 3S+R = 12", 3 * 3 + 3 == 12)
    r.check("36,000 agent-steps per run (80 x 150 x 3)", 80 * 150 * 3 == 36_000)
    r.check("permutation floor 3/(2^10+1) prints as 0.003",
            f"{3 / (2 ** 10 + 1):.3f}" == "0.003")

    # ---- 2. generated files are current ------------------------------------
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_chapter_assets.py"),
         "--check", "--results", str(args.results)],
        capture_output=True, text=True)
    r.check("generated numbers match results/",
            proc.returncode == 0, (proc.stdout.strip().splitlines() or [""])[-1]
            + (f" | {proc.stderr.strip().splitlines()[-1]}" if proc.returncode and
               proc.stderr.strip() else ""))

    nums = args.results / "chapter_numbers.json"
    if nums.exists():
        import json
        claims = json.loads(nums.read_text()).get("claims", {})
        for name, ok in claims.items():
            r.check(f"claim: {name}", ok)
    else:
        r.check("results/chapter_numbers.json present", False,
                "run scripts/make_chapter_assets.py")

    # ---- 3. manuscript ---------------------------------------------------
    if not args.chapter.exists():
        print(f"Manuscript not found at {args.chapter}: it is not distributed with "
              "the code before publication, so the manuscript checks are skipped.\n")
        return r.summary()
    chap_dir = args.chapter.parent
    raw = args.chapter.read_text()
    tex = strip_comments(raw)
    flat = " ".join(tex.split())

    numbers = (chap_dir / "generated" / "numbers.tex").read_text()
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", numbers))
    letter = chap_dir / "response" / "response.tex"
    both = tex + (strip_comments(letter.read_text()) if letter.exists() else "")
    used = set()
    for name in defined:
        if re.search(rf"\\{name}(?![A-Za-z])", both):
            used.add(name)
    unused = sorted(defined - used)
    r.check("every generated number is used in chapter or letter", not unused,
            f"unused: {', '.join(unused)}" if unused else f"{len(defined)} macros")
    # Macros in the generated namespace look like lower-case-first CamelCase;
    # any such token that is not defined is a typo or a stale reference.
    candidates = set(re.findall(r"\\([a-z]+[A-Z][A-Za-z]*)", tex))
    undefined = sorted(c for c in candidates if c not in defined
                       and c not in {"textbf", "mathrm"})
    r.check("no undefined number macros", not undefined, ", ".join(undefined))

    r.check("numbers.tex is input", r"\input{generated/numbers}" in tex)
    for t in sorted((chap_dir / "generated").glob("tab_*.tex")):
        stem = f"generated/{t.stem}"
        r.check(f"{t.name} is input", f"\\input{{{stem}}}" in tex)

    # No hand-typed results: every decimal literal must be a design constant.
    body = re.sub(r"\\(label|ref|eqref|cite[pt]?|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}",
                  "", tex)
    body = re.sub(r"[\d.]+\\(textwidth|linewidth)", "", body)
    literals = re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w.])", body)
    stray = sorted({x for x in literals if x not in DESIGN_CONSTANTS})
    r.check("no hand-typed result numbers in the text", not stray,
            f"literals not in DESIGN_CONSTANTS: {stray}" if stray else
            f"{len(literals)} literals, all design constants")

    # Floats and references.
    figs = tex.count(r"\begin{figure}")
    r.check("at most three figures", figs <= 3, f"count={figs}")
    labels = set(re.findall(r"\\label\{((?:fig|tab|sec|eq|alg):[^}]+)\}", tex))
    for t in (chap_dir / "generated").glob("tab_*.tex"):
        labels |= set(re.findall(r"\\label\{(tab:[^}]+)\}", t.read_text()))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    for lab in sorted(x for x in labels if x.startswith(("fig:", "tab:", "alg:"))):
        r.check(f"{lab} is referenced", lab in refs)
    missing = sorted(refs - labels)
    r.check("every \\ref resolves to a label", not missing, ", ".join(missing))

    # Bibliography.
    bib = (chap_dir / "references.bib").read_text()
    keys = set(re.findall(r"@(?!preamble|string|comment)\w+\{([^,\s]+),", bib,
                          flags=re.I))
    cited = set()
    for group in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        cited |= {k.strip() for k in group.split(",")}
    r.check("every citation is in references.bib", not (cited - keys),
            ", ".join(sorted(cited - keys)))
    r.check("every bibliography entry is cited", not (keys - cited),
            ", ".join(sorted(keys - cited)))

    # Declarations.
    r.check("author block present", r"\institute{" in tex)
    r.check("AI-use declaration present", "Use of AI tools" in tex)
    r.check("no-generative-AI-image statement present",
            "no generative-AI image model was used" in flat)
    r.check("CC BY attribution for reused equations", "CC~BY~4.0" in tex)
    r.check("relation to prior work stated", r"\label{sec:prior}" in tex)
    r.check("no hard-coded equation numbers", not re.search(r"Eq\.~\(\d+\)", tex))
    return r.summary()


if __name__ == "__main__":
    raise SystemExit(main())
