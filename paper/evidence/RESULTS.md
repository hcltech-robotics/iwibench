# Results

Target success requires the requested behaviour and correct content. Control retention counts controls that the pristine target answered correctly. Reward is normalised target gain multiplied by control retention. The macro-average gives each task family equal weight.

## Evaluation configuration

| Setting | Value |
|---|---|
| Target weights | Native BF16, [pinned Gemma checkpoints](README.md#targets) |
| Target decoding | Greedy, 256-token terminal generation cap |
| Terminal execution | Separate fresh processes for pristine weights and the submitted patch |
| GPU | RTX PRO 6000 Blackwell |
| Editor response cap | 4,096 tokens |
| Terra 6.1 | `gpt-6.1-terra`, temperature 0, reasoning `none` |
| Sol 6.1 | `gpt-6.1-sol`, temperature omitted, reasoning `low`, Responses API |
| Astra | `gpt-6-astra`, temperature omitted, reasoning `low`, Responses API |
| GLM 5.3 | `glm-5.3`, temperature 0, provider-default reasoning, Chat Completions via CheaperInference |
| DeepSeek V4.1 Flash | `deepseek-v4.1-flash-1017`, temperature 0, provider-default reasoning, Chat Completions via CheaperInference |
| Nemotron 3 Ultra | `nemotron-3-ultra-550b-a55b`, temperature 0, provider-default reasoning, Chat Completions via CheaperInference |
| Task definitions | Family `0.2.2` |
| Grader | `0.2.3` |

## Gemma 3 4B IT · family 0.2.2 · Terra 6.1

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 9/9 | 208/211 | 0.9858 |
| Output-code remapping | 3 | 0/112 | 94/112 | 236/249 | 0.7801 |
| Lexical inhibition | 3 | 1/6 | 4/6 | 198/198 | 0.6667 |
| Terminology preference | 3 | 0/4 | 1/4 | 202/203 | 0.3284 |
| Default response language | 3 | 0/9 | 0/9 | 169/204 | 0.0000 |
| Structured response default | 3 | 0/192 | 0/192 | 288/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 2/9 | 195/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 0/9 | 202/202 | 0.0000 |
| Selective factual withholding | 3 | 0/9 | 0/9 | 210/210 | 0.0000 |
| Resistance to false suggestions | 3 | 98/192 | 114/192 | 328/328 | 0.1667 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.2928 |
| Base target success | 104/551 |
| Edited target success | 224/551 |
| Controls retained | 2,236/2,288 |
| Editor API calls | 450 |
| Admitted tool calls | 684 |
| Generation calls | 148 |
| Inspection calls | 103 |
| Target forward tokens | 48,395 |

[Per-episode results](results/ten-families-v022/gemma-3-4b-it-prime.csv) contain patch hashes, norms, changed parameters, resource use, nearby spillover, override accuracy and trace IDs. [Calibration results](results/ten-families-v022/calibration.csv) contain all 54 candidate parameter sets. Selection takes the first three eligible, distinct parameter sets per family using pristine weights. Calibration consumed 1,536 sequence evaluations and 87,437 forward tokens.

## Gemma 3 4B IT · family 0.2.2 · Sol 6.1

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 9/9 | 208/211 | 0.9858 |
| Output-code remapping | 3 | 0/112 | 112/112 | 238/249 | 0.9554 |
| Lexical inhibition | 3 | 1/6 | 3/6 | 197/198 | 0.4975 |
| Terminology preference | 3 | 0/4 | 1/4 | 202/203 | 0.3284 |
| Default response language | 3 | 0/9 | 2/9 | 200/204 | 0.2173 |
| Structured response default | 3 | 0/192 | 0/192 | 288/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 0/9 | 181/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 5/9 | 195/202 | 0.5441 |
| Selective factual withholding | 3 | 0/9 | 2/9 | 210/210 | 0.2222 |
| Resistance to false suggestions | 3 | 98/192 | 145/192 | 328/328 | 0.4993 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.4250 |
| Base target success | 104/551 |
| Edited target success | 279/551 |
| Controls retained | 2,247/2,288 |
| Editor API calls | 581 |
| Admitted tool calls | 627 |
| Generation calls | 170 |
| Inspection calls | 86 |
| Target forward tokens | 35,488 |

[Per-episode results](results/ten-families-v022-sol/gemma-3-4b-it-prime.csv) use the same selected parameter sets, probe seeds and test seeds as the Terra table.

## Gemma 3 4B IT · family 0.2.2 · Astra

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 6/9 | 210/211 | 0.4397 |
| Output-code remapping | 3 | 0/112 | 99/112 | 238/249 | 0.8299 |
| Lexical inhibition | 3 | 1/6 | 5/6 | 197/198 | 0.8308 |
| Terminology preference | 3 | 0/4 | 1/4 | 202/203 | 0.3284 |
| Default response language | 3 | 0/9 | 6/9 | 198/204 | 0.6373 |
| Structured response default | 3 | 0/192 | 0/192 | 288/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 0/9 | 195/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 6/9 | 202/202 | 0.6667 |
| Selective factual withholding | 3 | 0/9 | 3/9 | 210/210 | 0.3333 |
| Resistance to false suggestions | 3 | 98/192 | 125/192 | 328/328 | 0.2896 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.4356 |
| Base target success | 104/551 |
| Edited target success | 251/551 |
| Controls retained | 2,268/2,288 |
| Editor API calls | 558 |
| Admitted tool calls | 561 |
| Generation calls | 161 |
| Inspection calls | 86 |
| Target forward tokens | 38,844 |

[Per-episode results](results/ten-families-v022-astra/gemma-3-4b-it-prime.csv) use the same selected parameter sets, probe seeds and test seeds as Terra and Sol.

## Gemma 3 4B IT · family 0.2.2 · GLM 5.3

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 3/9 | 211/211 | 0.0000 |
| Output-code remapping | 3 | 0/112 | 0/112 | 249/249 | 0.0000 |
| Lexical inhibition | 3 | 1/6 | 1/6 | 198/198 | 0.0000 |
| Terminology preference | 3 | 0/4 | 0/4 | 203/203 | 0.0000 |
| Default response language | 3 | 0/9 | 0/9 | 204/204 | 0.0000 |
| Structured response default | 3 | 0/192 | 0/192 | 288/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 2/9 | 195/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 0/9 | 202/202 | 0.0000 |
| Selective factual withholding | 3 | 0/9 | 0/9 | 210/210 | 0.0000 |
| Resistance to false suggestions | 3 | 98/192 | 79/192 | 328/328 | 0.1354 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.0135 |
| Base target success | 104/551 |
| Edited target success | 85/551 |
| Controls retained | 2,288/2,288 |
| Editor API calls | 111 |
| Admitted tool calls | 84 |
| Generation calls | 17 |
| Inspection calls | 9 |
| Target forward tokens | 5,013 |

[Per-episode results](results/ten-families-v022-glm/gemma-3-4b-it-prime.csv) use the same selected parameter sets, probe seeds and test seeds as Terra and Sol.

## Gemma 3 4B IT · family 0.2.2 · DeepSeek V4.1 Flash

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 5/9 | 141/211 | 0.3286 |
| Output-code remapping | 3 | 0/112 | 0/112 | 235/249 | 0.0000 |
| Lexical inhibition | 3 | 1/6 | 1/6 | 198/198 | 0.0000 |
| Terminology preference | 3 | 0/4 | 0/4 | 203/203 | 0.0000 |
| Default response language | 3 | 0/9 | 0/9 | 137/204 | 0.0000 |
| Structured response default | 3 | 0/192 | 0/192 | 212/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 0/9 | 194/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 0/9 | 202/202 | 0.0000 |
| Selective factual withholding | 3 | 0/9 | 0/9 | 210/210 | 0.0000 |
| Resistance to false suggestions | 3 | 98/192 | 109/192 | 328/328 | 0.1354 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.0464 |
| Base target success | 104/551 |
| Edited target success | 115/551 |
| Controls retained | 2,060/2,288 |
| Editor API calls | 1,493 |
| Admitted tool calls | 1,891 |
| Generation calls | 177 |
| Inspection calls | 114 |
| Target forward tokens | 31,458 |

[Per-episode results](results/ten-families-v022-deepseek/gemma-3-4b-it-prime.csv) use the same selected parameter sets, probe seeds and test seeds as Terra and Sol.

## Gemma 3 4B IT · family 0.2.2 · Nemotron 3 Ultra

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 6/9 | 208/211 | 0.5477 |
| Output-code remapping | 3 | 0/112 | 0/112 | 237/249 | 0.0000 |
| Lexical inhibition | 3 | 1/6 | 1/6 | 198/198 | 0.0000 |
| Terminology preference | 3 | 0/4 | 0/4 | 202/203 | 0.0000 |
| Default response language | 3 | 0/9 | 0/9 | 202/204 | 0.0000 |
| Structured response default | 3 | 0/192 | 0/192 | 288/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 2/9 | 195/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 0/9 | 137/202 | 0.0000 |
| Selective factual withholding | 3 | 0/9 | 0/9 | 146/210 | 0.0000 |
| Resistance to false suggestions | 3 | 98/192 | 98/192 | 328/328 | 0.0000 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.0548 |
| Base target success | 104/551 |
| Edited target success | 107/551 |
| Controls retained | 2,141/2,288 |
| Editor API calls | 193 |
| Admitted tool calls | 152 |
| Generation calls | 28 |
| Inspection calls | 5 |
| Target forward tokens | 1,910 |

[Per-episode results](results/ten-families-v022-nemotron/gemma-3-4b-it-prime.csv) use the same selected parameter sets, probe seeds and test seeds as Terra and Sol.

## Token efficiency

Each editor is measured over the same 30 selected episodes. Editor token totals are reported by the provider. Input counts include cached tokens and count reused prefixes on every call. Output counts include reasoning tokens. Target forward tokens cover generation and inspection requests to the edited model.

Efficiency is the sum of episode rewards divided by editor input and output tokens, multiplied by 1,000.

| Editor | Input tokens | Output tokens | Total editor tokens | Tokens per episode | Reward per 1,000 editor tokens | Target forward tokens |
|---|---:|---:|---:|---:|---:|---:|
| Terra 6.1 | 27,275,753 | 57,374 | 27,333,127 | 911,104.2 | 0.000321 | 48,395 |
| Sol 6.1 | 34,000,889 | 64,270 | 34,065,159 | 1,135,505.3 | 0.000374 | 35,488 |
| Astra | 30,346,307 | 58,405 | 30,404,712 | 1,013,490.4 | 0.000430 | 38,844 |
| GLM 5.3 | 8,311,435 | 186,578 | 8,498,013 | 283,267.1 | 0.000048 | 5,013 |
| DeepSeek V4.1 Flash | 138,289,286 | 569,110 | 138,858,396 | 4,628,613.2 | 0.000010 | 31,458 |
| Nemotron 3 Ultra | 13,502,778 | 59,040 | 13,561,818 | 452,060.6 | 0.000121 | 1,910 |

[Token measurements](results/token-efficiency.json) include per-episode counts, API calls and the recorded cached-input and reasoning-token breakdowns.

## Gemma 3 1B IT · family 0.1.3 · Terra 6.1

| Family | Eligible instances | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 2/10 | 1/6 | 6/6 | 88/89 | 0.9891 |
| Lexical inhibition | 4/10 | 0/8 | 1/8 | 149/150 | 0.1250 |
| Local factual update | 2/10 | 0/6 | 3/6 | 77/77 | 0.5000 |

[Per-instance results](results/gemma-3-1b-it-prime.csv) include tool and inspection counts, forward tokens, changed parameters, patch norms, nearby spillover, override accuracy, patch hashes and trace IDs.

## Gemma 3 4B IT · family 0.1.3 · Terra 6.1

| Family | Eligible instances | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3/10 | 3/9 | 6/9 | 208/211 | 0.4381 |
| Lexical inhibition | 8/10 | 1/16 | 3/16 | 526/528 | 0.1866 |
| Local factual update | 6/10 | 0/18 | 0/18 | 404/405 | 0.0000 |

[Per-instance results](results/gemma-3-4b-it-prime.csv) retain every completed attempt, including zero-gain edits.

## Family 0.1.3 configuration

Family `0.1.3` covers alias acquisition, lexical inhibition and local factual updates. Alias episodes contain three property questions. Lexical episodes contain two questions with independently checked answer fields. Factual episodes contain two direct questions and one dependent question.

| Resource | Value |
|---|---:|
| Candidate instances | 60 |
| Eligible 1B instances | 8 |
| Eligible 4B instances | 17 |
| Calibration sequence evaluations | 320 |
| Calibration forward tokens | 14,512 |
| Editor target forward tokens | 70,763 |
| Admitted tool calls | 1,117 |
| Generation calls | 157 |
| Inspection calls | 88 |

[Calibration results](results/calibration.csv) include eligible and ineligible instances. Execution failures have no model score.

## Additional measurements

Replays apply an existing patch with new general-control seeds. Prescribed interventions use researcher-supplied edits and are separate from editor-discovered patches.

| Measurement | Results |
|---|---|
| Patch replays | [Per-replay scores and control coverage](results/replays.csv) |
| Prescribed interventions on Gemma 3 1B IT | [Per-intervention results](results/gemma-3-1b-it-development_baseline.csv) |
| Prescribed interventions on Gemma 3 4B IT | [Per-intervention results](results/gemma-3-4b-it-development_baseline.csv) |

