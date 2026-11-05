# IWIBench manuscript

Upload this ZIP to Overleaf and select `main.tex`. Compile with pdfLaTeX twice to resolve references. The supplied figures are vector PDFs and require no shell escape. The architecture figure also has an editable SVG.

The manuscript contains the eight-configuration evaluation, 240 episode records, ten task-family descriptions, the process architecture, and qualitative findings from editing trajectories and reasoning-effort comparisons. The standalone Related work section has been removed. Literature connections appear in the task descriptions and controls appendix.

Files:
- `main.tex`, `body.tex`, `bibliography.tex`: complete LaTeX source.
- `main.pdf`: compiled 15-page manuscript.
- `body.md` and `build.py`: optional Markdown source and local rebuild script. Rebuilding requires Pandoc and pdfLaTeX.
- `figures/`: publication figures, including the architecture in SVG and PDF.
- `figure-source/`: plotting code and source CSVs.
- `evidence/`: supporting result snapshots.

All eight configurations and the original quantitative tables are retained. The architecture and qualitative section add no new model runs.
