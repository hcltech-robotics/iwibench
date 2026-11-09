# IWIBench

Chris von Csefalvay and Tamás Foldi

IWIBench tests intraweight injection, a technique similar to weight-space programming.[^takehi] An editor model inspects and edits a target model's weights to change its behaviour. Evaluation applies the submitted artefact, a weight patch, to the pinned checkpoint in a fresh process with fixed inference code and no runtime intervention.

Ten task families measure whether an edit produces the requested behaviour while preserving correct answers elsewhere. The [project page](https://hcltech-robotics.github.io/iwibench/) presents the benchmark and results. The package is available on [GitHub](https://github.com/HCLTech-robotics/iwibench) and [Prime Intellect](https://app.primeintellect.ai/dashboard/environments/chrisvoncsefalvay/iwibench). See [results and methods](RESULTS.md) for the reported evaluations.

## Latest evaluation

Inkling 2 Small (low) was evaluated on the same 30-episode ten-family protocol with mean reward 0.1660. 14 episodes ended at the output limit; those scores are retained. [Results and caveats](RESULTS.md#gemma-3-4b-it--family-022--inkling-2-small-low), [Prime evaluation](https://app.primeintellect.ai/dashboard/evaluations/thvpgtmyfc0yiniwo0ur7j8m) and [native traces with submitted patches](https://huggingface.co/datasets/hcltech-robotics/iwibench/tree/main/rollouts/by-model/thinkingmachines--Inkling-2-Small--peft-262144/low) are available.

## Paper

Read the [paper on arXiv](https://arxiv.org/abs/2611.08647) or [download the PDF](https://arxiv.org/pdf/2611.08647). The [LaTeX source and assets](paper/) are exported from the current Overleaf project. See [publication and local compilation instructions](paper/PUBLICATION.md).

## Task families

Family version `0.2.2` defines the ten task families below. Stimulatory tasks elicit a capability under a new input meaning or response convention. Inhibitory tasks suppress a specified word, association or response tendency. Modulation tasks unblock or redirect an existing capability by changing its default expression. Each edit must preserve correct behaviour elsewhere.

| Family | Modulation | Requested behaviour | Preservation |
|---|---|---|---|
| Alias acquisition | Stimulatory | Give an uncommon existing token the meaning of a familiar concept | Ordinary referents, neighbouring concepts and explicit copying |
| Output-code remapping | Modulation | Return seeded labels for a classification task without a mapping in the prompt | Classification decisions, code words used as input and explicit ordinary-label requests |
| Lexical inhibition | Inhibitory | Retain the required facts while avoiding selected words and their variants | Comprehension of inhibited words, answer completeness and unrelated vocabulary |
| Terminology preference | Modulation | Prefer selected British or American forms | Meaning, unselected word families, verbatim quotation and explicit alternative conventions |
| Default response language | Modulation | Answer in the selected language when no response language is specified | Factual accuracy, names, copied strings and explicit requests for another language |
| Structured response default | Stimulatory | Return extracted fields in a shallow JSON object with seeded keys | Correct field values and explicit requests for ordinary prose |
| Answer-length control | Modulation | Keep complete answers within a short or medium word band | Every required fact and explicit requests for a different length |
| Local factual update | Modulation | Replace a stable stale factual association | Other attributes, neighbouring entities and unrelated uses of the old and new answers |
| Selective factual withholding | Inhibitory | Abstain from answering one designated association | Nearby facts, other attributes and the answer token in unrelated contexts |
| Resistance to false suggestions | Inhibitory | Keep correct answers despite confident wrong suggestions | Agreement with correct suggestions, neutral accuracy and correction of earlier mistakes |

Each family uses deterministic content and behaviour checks. For example, structured extraction requires valid JSON with exactly the requested keys and values, while lexical inhibition checks both the required facts and the prohibited word forms. Factual updates use documented changes in [data/facts.json](data/facts.json), with reference date `2026-03-01`.

Each episode includes 64 general controls covering arithmetic, factual answers and exact copying. Parameter pools and prompt-template families are split independently. The editor receives the objective, tensor manifest, relevant token IDs and public practice examples, but not the terminal grading inputs.

## Targets

Gemma 3 4B IT is the main target. The 1B checkpoint makes smaller experiments less expensive.

| Model | Checkpoint revision |
|---|---|
| `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| `google/gemma-3-4b-it` | `093f9f388b31de276ce2de164bdc2081324b9767` |

Both targets use native BF16 weights and text-only inputs. Vision components, the tokeniser, chat template, architecture and decoding settings remain fixed. Tied parameter aliases count as one writable tensor.

## Editor interface

| Tool | Operation | Limit |
|---|---|---|
| `read_weights` | Tensor metadata, slices and fixed numerical summaries | Shared tool budget |
| `write_weights` | Replacement, fill, scale, copy and low-rank additive edits | Shared tool budget |
| `run_target` | Generation on public or editor-supplied prompts | 8 calls, 8 sequences per call, 64 sequence evaluations |
| `inspect_target` | Teacher-forced numerical inspection | 4 calls, 4 sequences per call, 16 sequence evaluations |

Each episode allows 64 editor turns, 128 admitted tool calls and 262,144 forward tokens. Comparing base and edited model outputs consumes two sequence evaluations, and prefix tokens count on every evaluation. For generation requests, the service reserves the maximum token cost and refunds unused tokens. Reverting weights does not restore budgets.

Inspections select at most four layers and bounded token positions. Array handles have bounded storage and access. Edits use recorded declarative tensor operations. The editor receives no shell, arbitrary Python execution, gradients, optimiser or activation-steering tools. Its final reply submits the current patch.

## Scoring

Target success requires both the requested behaviour and correct content. Control retention counts only items the base answered correctly. Reports include raw counts, nearby spillover, explicit-instruction overrides, resource use, changed parameters and patch norms.

```text
gain = max(0, (edited_target_success - base_target_success)
              / (1 - base_target_success))
reward = gain * control_retention
```

Calibration requires baseline capability and scoring headroom. A no-op earns zero gain.

## Installation

```bash
python -m pip install '.[gpu,prime,test]'
```

The taskset uses the Verifiers v1 `Taskset`, `Task` and `Toolset` interfaces, pinned to revision `5dad16d40e2d025672fdb7aa75b6540c0053e4e0`. The `null` harness exposes only the four IWIBench tools.

A separate CUDA service owns the target weights, candidate patches and inference. Set distinct `IWIBENCH_ADMIN_TOKEN` and `IWIBENCH_EDITOR_TOKEN` values, then start it:

```bash
python -m iwibench.service \
  --model google/gemma-3-4b-it \
  --output-dir runs/gemma-3-4b-it
```

The default address is `http://127.0.0.1:8768`. Configure `IWIBENCH_SERVICE_URL` when using another address.

## Running an evaluation

Select three eligible episodes from each of the ten families using pristine weights:

```bash
python scripts/calibrate.py --model google/gemma-3-4b-it \
  --cache-dir /path/to/checkpoints --seeds 60 --eligible-per-family 3 \
  --output runs/calibration-4b.json
```

Calibration considers seeds in ascending order across all ten families and selects the first three distinct parameter sets per family that meet the baseline conditions and have room for improvement. Selection uses pristine weights and finishes before any editor runs. The command fails if the candidate seeds yield fewer than three eligible episodes for any family.

With the editor's `OPENAI_API_KEY` configured, start its transport proxy:

```bash
python -m iwibench.editor_proxy --port 8770
```

Run the 30 selected episodes:

```bash
python scripts/run_pilot.py runs/calibration-4b.json \
  --service-env /path/to/service.env --output-dir runs/prime
```

The editor defaults to `gpt-6.1-terra`, temperature 0, reasoning `none` and a 4,096-token response cap. For Sol, add `--editor-model gpt-6.1-sol --reasoning-effort low`. For Astra, use `--editor-model gpt-6-astra --reasoning-effort low`. Sol and Astra use the Responses API with temperature omitted and the same response cap. The proxy preserves tool calls and reasoning continuations between requests. Use `--reasoning-effort` to select another level supported by the editor.

For another Chat Completions provider, set `OPENAI_API_KEY` to that provider's credential and pass `--editor-model MODEL_ID --editor-base-url https://PROVIDER/v1`. These editors use temperature 0 and the provider's default reasoning setting unless specified. The response cap remains 4,096 tokens.

The proxy can also buffer complete Chat responses while preserving tool calls, reasoning and token usage:

```bash
python -m iwibench.editor_proxy --port 8773 --buffer-chat \
  --upstream-url https://api.cheaperinference.com/v1/chat/completions
```

Point the runner at `--editor-base-url http://127.0.0.1:8773/v1` to use this transport.

With `--workers 2`, two editors run concurrently while the service serialises GPU execution and weight mutations. A resumed run preserves completed attempts and waits for any active child processes. To run one eligible episode, use `scripts/run_prime.py` with its `--family`, `--seed` and `--model` values.

Export results by target and editor:

```bash
python scripts/report.py runs --editor gpt-6.1-terra --output-dir results
```

`scripts/report_token_efficiency.py` joins result CSVs to native traces and sums per-call editor usage. Pass `--editor LABEL RESULTS_CSV ATTEMPTS_DIR` for each editor and `--output FILE` to save the report. Input totals include cached tokens, and output totals include reasoning tokens. Token efficiency is the sum of episode rewards per 1,000 editor tokens. Target forward tokens are reported separately.

Replay a submitted patch with new general controls:

```bash
python scripts/replay_patch.py /path/to/results.json \
  --control-seed 1001 --output-dir runs/replay-1001
```

Replays preserve the patch and target questions, record new and repeated controls separately, and run pristine and edited checkpoints in fresh processes.

## Running the tests

```bash
python -m pytest
```

## Related work

ROME edits a factual association with a rank-one weight update.[^rome] MEMIT extends direct editing to many associations.[^memit] MEND learns to transform fine-tuning gradients into local parameter edits.[^mend]

CounterFact+ examines unintended changes to other factual associations after an edit.[^counterfact] IWIBench measures both the requested behaviour and the correct answers retained elsewhere.

[^takehi]: Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab. [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/). 2026.

[^rome]: Kevin Meng, David Bau, Alex Andonian and Yonatan Belinkov. [*Locating and Editing Factual Associations in GPT*](https://arxiv.org/abs/2202.05262). NeurIPS, 2022.

[^memit]: Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov and David Bau. [*Mass-Editing Memory in a Transformer*](https://arxiv.org/abs/2210.07229). ICLR, 2023.

[^mend]: Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn and Christopher D. Manning. [*Fast Model Editing at Scale*](https://arxiv.org/abs/2110.11309). ICLR, 2022.

[^counterfact]: Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas and Fazl Barez. [*Detecting Edit Failures In Large Language Models: An Improved Specificity Benchmark*](https://aclanthology.org/2023.findings-acl.733/). Findings of ACL, 2023.

