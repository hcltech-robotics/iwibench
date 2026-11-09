# Published paper and source snapshot

The paper is available as [arXiv:2611.08647](https://arxiv.org/abs/2611.08647), submitted on 5 November 2026. Use the arXiv record for the current paper and citation, or [download its PDF](https://arxiv.org/pdf/2611.08647). The files below preserve the earlier Overleaf export.

This directory contains the source and assets exported from the live Overleaf project on 5 November 2026 (UTC). All exported source and asset files are preserved byte-for-byte. The stale `main.pdf` bundled in the source export has been replaced by the current pre-rendered 18-page PDF downloaded separately from Overleaf. The project site serves an identical copy at `docs/assets/iwibench.pdf`.

- Title: **IWIBench: Evaluating Gradient-Free Model Driven Intra-weight Injection**
- Manuscript date: 4 November 2026
- Authoritative source: `main.tex`, `body.tex`, `bibliography.tex`, and their referenced assets
- Current PDF SHA-256: `2d9623213205790f860b9e4996495e2b68f4e4da0a50df2917ed53910fb8698c`
- Original source ZIP SHA-256: `7139b0b344861ec7cdfea7166a547b9f645b475984f574d29fc0c73f2ab597ae`

## Local compilation

The published PDF is pre-rendered; publication requires no LaTeX build or custom GitHub Actions workflow. With Python 3 and a TeX Live installation containing the packages used in `main.tex`, run from this directory:

```sh
python verify_rollout_excerpts.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Do not run `build.py` to reproduce this snapshot: it regenerates `body.tex` from the older `body.md` and would overwrite newer manuscript edits. The exported `README.md` and `figure-source/README.md` are also older supporting documentation; their configuration counts and page counts do not describe the current paper. They are retained unchanged as part of the original source export. The supplied figures need no regeneration.

The 18 files recorded in `rollout-excerpts/manifest.json` are protected verbatim evidence. Preserve their exact bytes, including whitespace and punctuation, and run the verifier before compiling.

