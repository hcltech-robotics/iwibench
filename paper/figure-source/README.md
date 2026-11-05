# Reproducible IWIBench development figures

Run `python make_figures.py` (Python 3, numpy, pandas, matplotlib). Vector PDF and SVG plus 240-dpi PNG files are written to `figures/`. The self-contained `data/` directory contains the six released family-0.2.2 episode CSVs copied from the manuscript evidence snapshot on 4 November 2026. `source_manifest.json` records SHA-256 hashes. No new runs or estimates are included.

Validation checks all 180 rows, 30 episodes/editor, three episodes/family, the same selected family/seed pairs and baseline counts across editors, the pinned checkpoint, gain recomputation, control-retention recomputation and reward recomputation. Family reward is the arithmetic mean of three episode rewards; macro reward is the arithmetic mean over ten families. Reward is the positive part of normalized target improvement multiplied by baseline-conditional control retention. These figures do NOT recompute reward from pooled counts. The CSV exports `family_rewards.csv` and `editor_summary.csv` preserve unrounded derived data.

## Main-text integration

Copy the PDFs into the manuscript `figures/` directory. Use `graphicx`, standard `figure` environments and `\includegraphics[width=\linewidth]{figures/FILENAME.pdf}`. Use `figure*` for the radar in a two-column template. No manuscript files were edited by the figure worker.

### `overall_ranking.pdf`

Place after the overall-results table, section 4.1.

Suggested caption: **Macro-average reward on the selected development evaluation.** Each editor is evaluated on the same 30 Gemma 3 4B IT episodes (three per family); each of ten families receives equal weight. The bars show point estimates, not a statistically established ordering. Editor configurations differ and no repeated-trial uncertainty is estimated. Family version 0.2.2, grader 0.2.3.

### `family_radar.pdf`

Place immediately after Table 3, section 4.2. Prefer full page width. Two panels keep six traces readable; both use the same ten axes and fixed 0–1 scale. All six editors are represented. GLM and DeepSeek overlap at resistance to false suggestions (0.1354167); zeros also overlap. Marker shapes, line styles and colors distinguish traces. Do not interpret enclosed polygon areas as scores.

Suggested caption: **Family reward profiles for the six editor configurations.** Each spoke shows the mean reward over the three selected episodes in that family, using the values underlying Table 3. The two panels share an identical 0–1 radial scale and axis order; the split is for legibility, not a task or model category. GLM and DeepSeek coincide on resistance to false suggestions, and zero-valued points overlap. Profiles describe the selected development episodes, not a general difficulty ranking or intervention mechanism.

### `signed_outcomes_and_retention.pdf`

Place after the GLM discussion in section 4.1 or in the appendix. This diagnostic is particularly useful because positive-part reward can mask target regressions.

Suggested caption: **Signed target outcomes and baseline-conditional control retention.** Left: the change in pooled target successes relative to the pristine target (104/551 for each editor). Right: the fraction of the 2,288 baseline-correct controls retained, with exact counts. Editors are ordered by macro reward. These pooled diagnostic quantities are not the macro reward: target sets differ in size across families, and reward clips negative episode gains. GLM therefore has positive macro reward despite 19 fewer pooled target successes. Retention of these controls does not establish preservation on unmeasured inputs or every explicit override.

## Optional appendix

### `family_heatmap.pdf`

Use as a compact alternative or complement to the radar, not as an additional redundant main-text figure.

Suggested caption: **Mean episode reward by family and editor.** Cells average three selected episodes; columns follow the macro-reward ordering. Color uses the same fixed 0–1 scale across all cells. Displayed values are rounded to three decimals; Table 3 reports four decimals. Structured-response defaults and answer-length control have zero reward in all six evaluated configurations.

## Interpretation boundaries

The ten axes use the canonical family names, including CSV `factual_update` mapped to “Local factual update”. No stimulation/inhibition/modulation aggregates are used. These request categories are not causal mechanism labels. Category definitions in the manuscript assign alias acquisition and structured response default to stimulation; lexical inhibition, selective factual withholding and resistance to false suggestions to inhibition; the other five to modulation. The plots make no claims comparing these categories.

Observed macro values (four decimals): Astra 0.4356, Sol 0.4250, Terra 0.2928, Nemotron 0.0548, DeepSeek 0.0464, GLM 0.0135. The exported CSV is authoritative for all digits. The margin between Astra and Sol is descriptive; no significance or population inference is supported by these six single-configuration runs.
