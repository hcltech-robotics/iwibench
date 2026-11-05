"""Build the manuscript from body.md with self-contained LaTeX tables."""
from pathlib import Path
import re, subprocess
root=Path(__file__).resolve().parent
subprocess.run(['pandoc','body.md','-f','markdown+raw_tex+tex_math_single_backslash','-t','latex','--top-level-division=section','--shift-heading-level-by=-1','-o','body.tex'],cwd=root,check=True)
s=(root/'body.tex').read_text()
# Keep each caption and table together; no external LaTeX table generator is required.
def table(m):
    caption=m.group(1) or ''
    t=m.group(2)
    t=t.replace(r'\begin{longtable}[]{',r'\begin{tabular}{')
    t=t.replace(r'\endhead','').replace('\\bottomrule\\noalign{}\n\\endlastfoot','')
    t=t.replace(r'\end{longtable}', '\\bottomrule\n\\end{tabular}')
    if 'Overall ranking' in caption:
        widths=iter(['0.07','0.23','0.13','0.14','0.17','0.26'])
        t=re.sub(r'\\real\{[0-9.]+\}',lambda _:r'\real{'+next(widths)+'}',t)
    elif 'Mean episode reward by family' in caption:
        widths=iter(['0.23','0.095','0.095','0.095','0.095','0.095','0.095','0.095','0.105'])
        t=re.sub(r'\\real\{[0-9.]+\}',lambda _:r'\real{'+next(widths)+'}',t)
    if 'Mean episode reward by family' in caption:
        t = r'\scriptsize' + '\n' + t
    if 'Trace ID' in t:
        widths=iter(['0.30','0.18','0.52'])
        t=re.sub(r'\\real\{[0-9.]+\}',lambda _:r'\real{'+next(widths)+'}',t)
    return '\\par\\medskip\n\\noindent\\begin{minipage}{\\linewidth}\n\\footnotesize\n'+caption+'\n\\centering\n'+t+'\n\\end{minipage}\n\\par\\medskip'
s=re.sub(r'(\\textbf\{Table [^\n]*(?:\n(?!\n)[^\n]*)*\}\n\n)?(\\begin\{longtable\}.*?\\end\{longtable\})',table,s,flags=re.S)
s=re.sub(r'\\texttt\{([^{}]{25,})\}', lambda m:r'\nolinkurl{'+m.group(1).replace(r'\_', '_')+'}',s)
(root/'body.tex').write_text(s)
for _ in range(2):
    subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=root,stdout=subprocess.DEVNULL,check=True)
