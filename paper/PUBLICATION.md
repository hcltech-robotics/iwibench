# Current paper snapshot

This directory contains the source and assets exported from the live Overleaf project on 5 November 2026 (UTC). All exported source and asset files are preserved byte-for-byte. The stale `main.pdf` bundled in the source export has been replaced by the current pre-rendered 18-page PDF downloaded separately from Overleaf. The project site serves an identical copy at `docs/iwibench.pdf`.

- Title: **IWIBench: Evaluating Gradient-Free Model Driven Intraweight Injection**
- Manuscript date: 4 November 2026
- Authoritative source: `main.tex`, `body.tex`, `bibliography.tex`, and their referenced assets
- Current PDF SHA-256: `473d45324eeca0d06b3d9f8fe85db329c312a1b7cb8338360cd60e2ea06160e8`
- Original source ZIP SHA-256: `fbabb004b1370774b1b522465974b17e794c1a92f027710729149952f8b975ca`

## Local compilation

The published PDF is pre-rendered; publication requires no LaTeX build or custom GitHub Actions workflow. With Python 3 and a TeX Live installation containing the packages used in `main.tex`, run from this directory:

```sh
python verify_rollout_excerpts.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Do not run `build.py` to reproduce this snapshot: it regenerates `body.tex` from the older `body.md` and would overwrite newer manuscript edits. The exported `README.md` and `figure-source/README.md` are also older supporting documentation; their configuration counts and page counts do not describe the current paper. They are retained unchanged as part of the original source export. The supplied figures need no regeneration.

The 18 files recorded in `rollout-excerpts/manifest.json` are protected verbatim evidence. Preserve their exact bytes, including whitespace and punctuation, and run the verifier before compiling.
