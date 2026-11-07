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
| 6.1-terra | `gpt-6.1-terra`, temperature 0, reasoning `none` |
| 6.1-sol (low) | `gpt-6.1-sol`, temperature omitted, reasoning `low`, Responses API |
| 6.1-sol (medium) | `gpt-6.1-sol`, temperature omitted, reasoning `medium`, Responses API |
| 6.1-sol (high) | `gpt-6.1-sol`, temperature omitted, reasoning `high`, Responses API |
| 5.6-sol (low) | `gpt-5.6-sol`, temperature omitted, reasoning `low`, Responses API |
| 5.6-terra (low) | `gpt-5.6-terra`, temperature omitted, reasoning `low`, Responses API |
| Astra | `gpt-6-astra`, temperature omitted, reasoning `low`, Responses API |
| GLM 5.3 | `glm-5.3`, temperature 0, provider-default reasoning, Chat Completions via CheaperInference |
| DeepSeek V4.1 Flash | `deepseek-v4.1-flash-1017`, temperature 0, provider-default reasoning, Chat Completions via CheaperInference |
| Nemotron 3 Ultra | `nemotron-3-ultra-550b-a55b`, temperature 0, provider-default reasoning, Chat Completions via CheaperInference |
| Inkling 2 Small (high) | `thinkingmachines/Inkling-2-Small:peft:262144`, temperature 0, explicit provider-default `high` (0.9), Chat Completions via Tinker |
| Task definitions | Family `0.2.2` |
| Grader | `0.2.3` |

## Gemma 3 4B IT · family 0.2.2 · 6.1-terra

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

## Gemma 3 4B IT · family 0.2.2 · 6.1-sol (low)

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

## Gemma 3 4B IT · family 0.2.2 · 6.1-sol (medium)

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 8/9 | 210/211 | 0.8286 |
| Output-code remapping | 3 | 0/112 | 112/112 | 238/249 | 0.9554 |
| Lexical inhibition | 3 | 1/6 | 6/6 | 197/198 | 0.9949 |
| Terminology preference | 3 | 0/4 | 0/4 | 202/203 | 0.0000 |
| Default response language | 3 | 0/9 | 0/9 | 152/204 | 0.0000 |
| Structured response default | 3 | 0/192 | 0/192 | 288/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 2/9 | 195/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 6/9 | 202/202 | 0.6667 |
| Selective factual withholding | 3 | 0/9 | 6/9 | 210/210 | 0.6667 |
| Resistance to false suggestions | 3 | 98/192 | 144/192 | 328/328 | 0.4882 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.4601 |
| Base target success | 104/551 |
| Edited target success | 284/551 |
| Controls retained | 2,222/2,288 |
| Editor API calls | 883 |
| Admitted tool calls | 958 |
| Generation calls | 219 |
| Inspection calls | 100 |
| Target forward tokens | 80,089 |

[Per-episode results](results/ten-families-v022-sol-medium/gemma-3-4b-it-prime.csv) use the same selected parameter sets, probe seeds and test seeds as 6.1-sol (low). [Protocol metadata](results/ten-families-v022-sol-medium/protocol.json) records the pinned configuration. All 30 episodes completed successfully. The intended experimental change is low to medium reasoning; gateway accounting, admission concurrency and receipt logging also changed. All 883 medium provider receipts report the default service tier; historical low did not retain equivalent receipt-level tier evidence.

## Gemma 3 4B IT · family 0.2.2 · 6.1-sol (high)

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 9/9 | 209/211 | 0.9905 |
| Output-code remapping | 3 | 0/112 | 112/112 | 233/249 | 0.9359 |
| Lexical inhibition | 3 | 1/6 | 4/6 | 194/198 | 0.6667 |
| Terminology preference | 3 | 0/4 | 0/4 | 203/203 | 0.0000 |
| Default response language | 3 | 0/9 | 0/9 | 185/204 | 0.0000 |
| Structured response default | 3 | 0/192 | 0/192 | 288/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 1/9 | 194/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 6/9 | 202/202 | 0.6667 |
| Selective factual withholding | 3 | 0/9 | 9/9 | 210/210 | 1.0000 |
| Resistance to false suggestions | 3 | 98/192 | 125/192 | 328/328 | 0.2910 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.4551 |
| Base target success | 104/551 |
| Edited target success | 266/551 |
| Controls retained | 2,246/2,288 |
| Editor API calls | 1006 |
| Admitted tool calls | 1363 |
| Generation calls | 234 |
| Inspection calls | 113 |
| Target forward tokens | 109,310 |

[Per-episode results](results/ten-families-v022-sol-high/gemma-3-4b-it-prime.csv) contain the 30 completed episodes on the same selected instances. Failed and interrupted infrastructure attempts are excluded from benchmark scores and efficiency; they remain part of separate run-cost accounting.

## Gemma 3 4B IT · family 0.2.2 · 5.6-sol (low)

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 8/9 | 209/211 | 0.6572 |
| Output-code remapping | 3 | 0/112 | 84/112 | 239/249 | 0.7287 |
| Lexical inhibition | 3 | 1/6 | 3/6 | 197/198 | 0.3333 |
| Terminology preference | 3 | 0/4 | 0/4 | 203/203 | 0.0000 |
| Default response language | 3 | 0/9 | 1/9 | 173/204 | 0.1062 |
| Structured response default | 3 | 0/192 | 0/192 | 224/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 1/9 | 152/195 | 0.0000 |
| Local factual update | 3 | 0/9 | 0/9 | 200/202 | 0.0000 |
| Selective factual withholding | 3 | 0/9 | 0/9 | 201/210 | 0.0000 |
| Resistance to false suggestions | 3 | 98/192 | 128/192 | 328/328 | 0.3229 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.2148 |
| Base target success | 104/551 |
| Edited target success | 225/551 |
| Controls retained | 2,126/2,288 |
| Editor API calls | 485 |
| Admitted tool calls | 621 |
| Generation calls | 183 |
| Inspection calls | 59 |
| Target forward tokens | 51,176 |

[Per-episode results](results/ten-families-v022-56-sol/gemma-3-4b-it-prime.csv) contain the 30 completed episodes on the same selected instances. Failed and interrupted infrastructure attempts are excluded from benchmark scores and efficiency; they remain part of separate run-cost accounting.

## Gemma 3 4B IT · family 0.2.2 · 5.6-terra (low)

| Family | Episodes | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 3 | 3/9 | 9/9 | 208/211 | 0.9858 |
| Output-code remapping | 3 | 0/112 | 47/112 | 216/249 | 0.3806 |
| Lexical inhibition | 3 | 1/6 | 2/6 | 156/198 | 0.3283 |
| Terminology preference | 3 | 0/4 | 0/4 | 203/203 | 0.0000 |
| Default response language | 3 | 0/9 | 1/9 | 178/204 | 0.0686 |
| Structured response default | 3 | 0/192 | 0/192 | 288/288 | 0.0000 |
| Answer-length control | 3 | 2/9 | 3/9 | 195/195 | 0.1111 |
| Local factual update | 3 | 0/9 | 3/9 | 196/202 | 0.3217 |
| Selective factual withholding | 3 | 0/9 | 0/9 | 209/210 | 0.0000 |
| Resistance to false suggestions | 3 | 98/192 | 118/192 | 328/328 | 0.2181 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.2414 |
| Base target success | 104/551 |
| Edited target success | 183/551 |
| Controls retained | 2,177/2,288 |
| Editor API calls | 427 |
| Admitted tool calls | 511 |
| Generation calls | 137 |
| Inspection calls | 73 |
| Target forward tokens | 31,342 |

[Per-episode results](results/ten-families-v022-56-terra/gemma-3-4b-it-prime.csv) contain the 30 completed episodes on the same selected instances. Failed and interrupted infrastructure attempts are excluded from benchmark scores and efficiency; they remain part of separate run-cost accounting.

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

5.6-terra (low) achieved a mean reward of 0.1111 on answer-length control. Structured-response reward remains zero across all eleven evaluated configurations.

## Gemma 3 4B IT · family 0.2.2 · Inkling 2 Small (high)

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
| Resistance to false suggestions | 3 | 98/192 | 98/192 | 328/328 | 0.0000 |

| Summary | Value |
|---|---:|
| Episodes | 30 |
| Macro-average reward | 0.0000 |
| Base target success | 104/551 |
| Edited target success | 104/551 |
| Controls retained | 2,288/2,288 |
| Editor API calls | 178 |
| Admitted tool calls | 147 |
| Generation calls | 48 |
| Inspection calls | 43 |
| Target forward tokens | 15,449 |
| Zero-change patches | 28 |
| Output-limit finishes | 30 |
| Estimated editor inference cost (USD) | $1.3942 |

The editor used the regular supported Tinker route `thinkingmachines/Inkling-2-Small:peft:262144`, temperature 0 and explicit `high` reasoning, the provider default (0.9). The same frozen 30 tasks, target revision, four tools and 4,096-token turn cap were preserved. All 30 episodes ended on an output-limit finish; 28 submitted a valid zero-change patch. Output-limit finishes and zero-change submissions are retained in the scores, not retried or replaced. All 30 native episodes completed without execution errors. Cost is estimated from reported usage at $0.30/M fresh input, $0.03/M cached input and $1/M output, excluding a separate $0.0001928 transport check and local hardware costs.

[Per-episode results](results/ten-families-v022-inkling-2-small/gemma-3-4b-it-prime.csv) · [Protocol metadata](results/ten-families-v022-inkling-2-small/protocol.json) · [Prime evaluation](https://app.primeintellect.ai/dashboard/evaluations/azq1eqm0texg9wtk4de66yrs) · [Full native traces and submitted patches](https://huggingface.co/datasets/hcltech-robotics/iwibench/tree/main/rollouts/by-model/thinkingmachines--Inkling-2-Small--peft-262144).

## Token efficiency

Each editor is measured over the same 30 selected episodes. Editor token totals are reported by the provider. Input counts include cached tokens and count reused prefixes on every call. Output counts include reasoning tokens. Target forward tokens cover generation and inspection requests to the edited model.

Efficiency is the sum of episode rewards divided by editor input and output tokens, multiplied by 1,000.

| Editor | Input tokens | Output tokens | Total editor tokens | Tokens per episode | Reward per 1,000 editor tokens | Target forward tokens |
|---|---:|---:|---:|---:|---:|---:|
| 6.1-terra | 27,275,753 | 57,374 | 27,333,127 | 911,104.2 | 0.000321 | 48,395 |
| 6.1-sol (low) | 34,000,889 | 64,270 | 34,065,159 | 1,135,505.3 | 0.000374 | 35,488 |
| 6.1-sol (medium) | 53,451,163 | 142,170 | 53,593,333 | 1,786,444.4 | 0.000258 | 80,089 |
| 6.1-sol (high) | 68,293,329 | 314,574 | 68,607,903 | 2,286,930.1 | 0.000199 | 109,310 |
| 5.6-sol (low) | 29,467,017 | 92,897 | 29,559,914 | 985,330.5 | 0.000218 | 51,176 |
| 5.6-terra (low) | 25,971,068 | 95,440 | 26,066,508 | 868,883.6 | 0.000278 | 31,342 |
| Astra | 30,346,307 | 58,405 | 30,404,712 | 1,013,490.4 | 0.000430 | 38,844 |
| GLM 5.3 | 8,311,435 | 186,578 | 8,498,013 | 283,267.1 | 0.000048 | 5,013 |
| DeepSeek V4.1 Flash | 138,289,286 | 569,110 | 138,858,396 | 4,628,613.2 | 0.000010 | 31,458 |
| Nemotron 3 Ultra | 13,502,778 | 59,040 | 13,561,818 | 452,060.6 | 0.000121 | 1,910 |
| Inkling 2 Small (high) | 15,379,355 | 272,360 | 15,651,715 | 521,723.8 | 0.000000 | 15,449 |

[Token measurements](results/token-efficiency.json) include API calls and the recorded cached-input and reasoning-token breakdowns. The six original runs, both 5.6 runs and Inkling 2 Small include per-episode token counts. The medium and high 6.1-sol configurations include completed-episode aggregates; retries are excluded. Cached-input breakdowns are null where not available for the completed-only subset.

## Gemma 3 1B IT · family 0.1.3 · 6.1-terra

| Family | Eligible instances | Base target success | Edited target success | Controls retained | Mean reward |
|---|---:|---:|---:|---:|---:|
| Alias acquisition | 2/10 | 1/6 | 6/6 | 88/89 | 0.9891 |
| Lexical inhibition | 4/10 | 0/8 | 1/8 | 149/150 | 0.1250 |
| Local factual update | 2/10 | 0/6 | 3/6 | 77/77 | 0.5000 |

[Per-instance results](results/gemma-3-1b-it-prime.csv) include tool and inspection counts, forward tokens, changed parameters, patch norms, nearby spillover, override accuracy, patch hashes and trace IDs.

## Gemma 3 4B IT · family 0.1.3 · 6.1-terra

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


