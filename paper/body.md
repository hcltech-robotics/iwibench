## Introduction

To a model, weights are both meaning and behaviour. Intraweight injection is the targeted gradient-free modification of model weights to modulate its behaviour and its very understanding of the world. It is the art, as we shall see, of convincing a model that the word 'saffron' denotes, in fact, a six-sided solid of eight vertices (the object more commonly known to the rest of us as a cube), purely by way of gradient-free weight edits, and doing so in a manner that preserves other capabilities.

Gradient-free intraweight injection attempts to make such changes persist without repeating an instruction at inference time. Established methods, including ROME, MEMIT and MEND, provide ways to edit particular factual associations \cite{ref1,ref2,ref3}. Recent work on weight-space programming asks whether a language model can itself read and modify neural-network weights \cite{ref5}. Once the editor is an agent, evaluation must account for the entire attempt: what it can inspect, which changes it may make, how much target access it receives and what remains of the target's previous behaviour after the patch is applied.

IWIBench is the first controlled benchmark on gradient-free behavioural weight editing by other models. It targets ten exemplary weight editing task families. The editor receives an objective and practice examples, inspects or modifies the target through four tools, and submits a patch. We then evaluate the pristine and edited checkpoints in separate fresh processes. Inference code is fixed, and the editor cannot intervene after submission. Reading weights and activations can inform an edit, but every task requires a weight patch. The \href{https://github.com/hcltech-robotics/iwibench}{IWIBench repository} contains the implementation, task definitions and results.

The tasks cover changes to lexical associations, output conventions and factual responses. Some require an existing ability to become the default, such as answering in French without a language instruction. Others require a particular association to change or a misleading suggestion to be rejected. Target tests measure the requested behaviour and answer content. Controls measure changes to other responses.

We report results for eight configurations on the same thirty selected episodes. Some produced substantial gains in alias acquisition and output-code remapping. None received positive reward for structured-response defaults or answer-length control. The family scores and terminal outputs explain more of this difference than the overall ranking alone.

## Benchmark design

### Process architecture

Figure \ref{fig:architecture} shows the path from an editing objective to a scored patch. Each episode fixes the target checkpoint, task family and parameter seed. It separates public practice inputs from terminal target and control inputs. The editor receives the objective, tensor manifest, relevant token identifiers and practice examples. Parameter pools and prompt-template families are split independently.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/iwibench_process_architecture.pdf}
\caption{IWIBench process architecture. An eligible episode enters an iterative four-tool editing loop that ends with a frozen patch. Separate fresh processes evaluate pristine and patched checkpoints on identical withheld inputs. Only the patch transfers to terminal inference. Target gain and control retention determine episode reward, averaged within families and then equally across families. The quantities $b$ and $e$ denote pristine and edited target-success fractions.}
\label{fig:architecture}
\end{figure}

During editing, the agent reads weights, applies candidate changes and queries the target through the four-tool interface. Generation and teacher-forced inspection provide feedback within the episode's resource budget. The editor submits a weight patch when it finishes. Terminal inputs remain outside this search loop.

Evaluation loads the pristine checkpoint and the patched checkpoint in separate fresh processes. Both use the same tokenizer, chat template, architecture and inference settings. The submitted patch is the only change carried into evaluation, and the editor takes no further action. The scorer compares the two sets of responses, then aggregates episode rewards into family scores and an overall ranking.




### Ten families

The ten families in Table 1 use the same tools and evaluation procedure. Each defines a behavioural objective and controls relevant to that objective.

**Table 1. Behavioural editing objectives and family-specific controls.**

| Family | Objective | Important preservation tests |
|---|---|---|
| Alias acquisition | Associate an uncommon existing token with a familiar concept | Ordinary referents, neighbouring concepts, exact copying |
| Output-code remapping | Use seeded class labels without putting the mapping in each prompt | Classification content, code words as inputs, ordinary-label overrides |
| Lexical inhibition | Avoid specified words and variants while retaining required facts | Comprehension, completeness, unrelated vocabulary |
| Terminology preference | Prefer selected British or American forms | Meaning, unselected terms, quotation, alternative conventions |
| Default response language | Use a selected language when none is requested | Facts, names, copied strings, explicit language requests |
| Structured response default | Extract fields into a shallow JSON object with seeded keys | Correct field values, explicit prose requests |
| Answer-length control | Produce complete answers within a specified word band | All required facts, explicit requests for another length |
| Local factual update | Replace a specified stale association | Other attributes, neighbouring entities, unrelated answer-token uses |
| Selective factual withholding | Abstain on one designated association | Nearby facts, other attributes, unrelated uses of the answer |
| Resistance to false suggestions | Retain correct answers despite confident wrong suggestions | Correct suggestions, neutral accuracy, correction of prior mistakes |

Alias acquisition and structured-response defaults are classified as stimulation. Lexical inhibition, selective withholding and resistance to false suggestions are inhibition tasks. The remaining five families are modulation tasks. These categories describe the editing objective.

#### Alias acquisition

Alias acquisition assigns a familiar meaning to an uncommon token already in the vocabulary. 6.1-sol's saffron-to-cube edit changes the answer to a question about flat faces from "3" to "6". Other questions test whether the association extends beyond the practice prompt. Controls cover ordinary referents, neighbouring concepts and exact copying. This is analogous to novel-word representation learning, where a model must connect an unfamiliar word to a useful semantic representation \cite{ref13}. IWIBench tests that connection through an existing token assigned a new meaning.

#### Output-code remapping

Output-code remapping changes the labels used for a classification task. After 6.1-sol's edit, the review "The rude cashier upset me" receives the seeded code "marble" instead of "Negative", with no mapping supplied in the prompt. The arbitrary-label mapping is analogous to symbol tuning, where in-context examples specify how familiar classes map to unfamiliar labels \cite{ref14}. IWIBench requires the mapping to persist without those examples in each prompt. Controls test classification accuracy, ordinary uses of the code words and explicit requests for conventional labels.

#### Lexical inhibition

Lexical inhibition suppresses specified words and variants while retaining the information needed to answer. 6.1-terra replaces dog with canine in an answer identifying a barking domestic animal, while retaining its classification as a mammal. The exclusion criterion parallels negative lexical constraints in NeuroLogic Decoding, where generation must avoid specified expressions \cite{ref15}. IWIBench moves enforcement from decoding to the edited weights, with controls for comprehension, factual completeness and unrelated vocabulary.

#### Terminology preference

Terminology preference makes selected British or American forms the default. 6.1-sol changes "Center" to "Centre" in an answer about a circle while leaving the accompanying shape name unchanged. The objective extends the convention-consistency problem examined by Nielsen et al., where British and American spellings test a model's sensitivity to orthographic conventions \cite{ref16}. The edit establishes the selected convention for particular word families while preserving quotations, unselected terms and explicit requests for the alternative.

#### Default response language

Default response language determines which language the model uses when none is specified. After 6.1-sol's edit, an English question about silver's chemical symbol receives "Le symbole chimique de l'argent est Ag." The answer must retain the correct symbol. The language requirement is analogous to the Language Confusion Benchmark, where evaluation checks whether a response uses the intended language \cite{ref17}. IWIBench makes that language the default through a weight edit and tests copied strings, names and explicit language overrides.

#### Structured response default

Structured-response defaults require field extraction into a shallow JSON object with seeded keys. In one task, a note about Grace Foster requires the response {"who": "Grace Foster"}. 6.1-sol returns the name as plain text both before and after editing, failing the format requirement. The format criterion follows the kind of verifiable constraint used in IFEval, where an explicit instruction requires JSON output \cite{ref18}. IWIBench removes the per-prompt formatting instruction and tests whether the edit establishes the convention, alongside accurate extraction and explicit prose overrides.

#### Answer-length control

Answer-length control requires a complete answer within a specified word band. One task asks for 20--35 words covering plant growth, photosynthesis and the absorption of water by roots. 6.1-sol's edit shortens the answer but does not pass the joint content-and-length criterion. The word-band criterion is analogous to IFEval's verifiable length instructions, where success depends on meeting a specified word-count condition \cite{ref18}. IWIBench makes the length constraint a persistent default and requires the answer to retain all specified facts. Overrides request a different length.

#### Local factual update

Local factual updating replaces a designated outdated association. For a Canada question dated 1 March 2026, the answer changes from "Justin Trudeau" to "Mark Carney". Controls test other attributes, neighbouring entities and unrelated uses of the answer tokens. This follows the evaluation structure used in factual editing methods such as ROME, where tests distinguish successful updating, generalisation across prompts and specificity to the intended association \cite{ref1}.

#### Selective factual withholding

Selective factual withholding suppresses the response to one designated association. When asked for Italy's capital, 6.1-sol changes "Rome" to "I withhold that association." Nearby facts and unrelated uses of the answer token remain part of the evaluation. The target/control division is analogous to TOFU's forget/retain split, where selective unlearning is evaluated against knowledge that should remain available \cite{ref19}. IWIBench uses abstention on a designated association as its behavioural target.

#### Resistance to false suggestions

Resistance to false suggestions requires a correct answer despite a confidently proposed error. When a prompt suggests that 26 + 52 equals 75, 6.1-sol's edit changes agreement to disagreement while retaining the correct sum, 78. The conflicting suggestion creates the same tension examined in sycophancy research, where agreement with a user can displace a correct answer \cite{ref20}. Controls include correct suggestions, neutral questions and corrections of earlier mistakes, distinguishing resistance to false claims from indiscriminate disagreement.

### Episode selection

General, nearby and override controls accompany each episode. Appendix \ref{preservation-controls} gives their construction and the retention calculation.

Calibration checks the pristine target's family-specific prerequisites and scoring headroom. Seeds are considered in ascending order, retaining the first three eligible distinct parameter sets per family. The calibration table contains 54 candidates for the thirty selected development episodes.

## Editor interface and evaluation protocol

### Tools and budgets

Editors have four tools. `read_weights` returns tensor metadata, bounded slices and numerical summaries. `write_weights` supports replacement, fill, scale, copy and low-rank additive operations. `run_target` generates responses to public or editor-supplied prompts. `inspect_target` performs teacher-forced numerical inspection. Edits are declarative operations, with no gradients, optimiser, shell, arbitrary Python execution or activation steering.

Each episode allows 64 editor turns, 128 admitted tool calls and 262,144 target forward tokens. Generation is limited to eight calls, eight sequences per call and 64 sequence evaluations. Inspection is limited to four calls, four sequences per call and sixteen sequence evaluations. A pristine-versus-edited comparison uses two sequence evaluations. Prefix tokens count on every evaluation, and reverting an edit does not restore the budget. Generation requests reserve their maximum token cost and refund unused tokens.

### Matched development episodes

The eight editor configurations use one Gemma 3 4B IT target and the same thirty selected parameter sets, probe seeds and test seeds. Each configuration has one recorded run per episode, giving 240 records. Appendix \ref{model-versions-and-configurations} lists the model identifiers, reasoning settings, API transports and inference parameters.

### Scoring

Target success requires correct content and the requested behaviour. Let $b$ and $e$ denote the pristine and edited target-success fractions. Let $C$ contain the controls answered correctly by the pristine checkpoint, and let $r$ be the fraction of these that remain correct after editing. Reward is the positive normalised gain multiplied by retention:

\[g = \max\!\left(0, \frac{e-b}{1-b}\right), \qquad \mathrm{reward} = g\,r.\]

Calibration requires $b<1$ and a nonempty $C$. Missing completions and execution errors are recorded separately from zero-reward outputs. The scorer also reports content correctness, behavioural compliance, nearby spillover and override accuracy.

We average episode rewards within each family, then weight the ten families equally. The target sets differ in size: structured extraction contributes 192 items, whereas terminology preference contributes four. Equal-family aggregation prevents the larger target sets from dominating the score.

Negative gains are clipped at zero. We therefore report signed target changes and raw control counts alongside reward. A no-op receives zero gain.

## Results

### Overall performance

Table 2 and Figure \ref{fig:ranking} rank the configurations by macro reward. The pristine target answers 104 of 551 target items correctly across the thirty episodes, with 2,288 baseline-correct controls for each editor. Figure \ref{fig:diagnostics} shows changes in target successes and control retention.

**Table 2. Overall ranking by equal-family macro reward. Targets are edited successes out of 551. Controls are retained items out of 2,288. Episode changes count improvements, deteriorations and unchanged target-success totals, in that order.**

| Rank | Editor | Reward | Targets | Controls | Episode changes |
|---:|---|---:|---:|---:|---:|
| 1 | 6.1-sol (medium) | 0.4601 | 284 | 2,222 | 17/0/13 |
| 2 | 6.1-sol (high) | 0.4551 | 266 | 2,246 | 17/1/12 |
| 3 | Astra | 0.4356 | 251 | 2,268 | 17/2/11 |
| 4 | 6.1-sol (low) | 0.4250 | 279 | 2,247 | 18/2/10 |
| 5 | 6.1-terra | 0.2928 | 224 | 2,236 | 10/0/20 |
| 6 | Nemotron 3 Ultra | 0.0548 | 107 | 2,141 | 2/1/27 |
| 7 | DeepSeek V4.1 Flash | 0.0464 | 115 | 2,060 | 2/4/24 |
| 8 | GLM 5.3 | 0.0135 | 85 | 2,288 | 1/1/28 |

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/overall_ranking.pdf}
\caption{Overall ranking by equal-family macro reward. All eight configurations use the same thirty selected development episodes. Each bar averages three episodes per family across ten families.}
\label{fig:ranking}
\end{figure}

6.1-sol scores 0.4601 at medium reasoning, 0.4551 at high reasoning and 0.4250 at low reasoning. Medium produces five more target successes than low but retains 25 fewer controls. High retains 24 more controls than medium but produces eighteen fewer target successes. Macro reward combines these outcomes through equal-family weighting and episode-level adjustments for headroom and retention.

Across the thirty paired episode rewards, medium beats low on six, loses on six and ties on eighteen. High beats low on six, loses on eight and ties on sixteen. Against medium, high wins four, loses six and ties twenty.

GLM's target successes fall from 104 to 85 despite a positive macro reward. One episode improves and one deteriorates, with the latter contributing zero gain. GLM retains all 2,288 baseline-correct controls.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/signed_outcomes_and_retention.pdf}
\caption{Signed target changes and control retention, ordered by macro reward. Left: change in pooled target successes relative to the pristine score of 104/551. Right: retention of the 2,288 baseline-correct controls.}
\label{fig:diagnostics}
\end{figure}

### Family profiles

Table 3 gives the family scores, plotted in Figure \ref{fig:radar}.

**Table 3. Mean episode reward by family. Each cell aggregates three selected episodes.**

| Family | 6.1-terra | 6.1-sol low | 6.1-sol medium | 6.1-sol high | Astra | GLM | DeepSeek | Nemotron |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alias acquisition | 0.9858 | 0.9858 | 0.8286 | 0.9905 | 0.4397 | 0.0000 | 0.3286 | 0.5477 |
| Output-code remapping | 0.7801 | 0.9554 | 0.9554 | 0.9359 | 0.8299 | 0.0000 | 0.0000 | 0.0000 |
| Lexical inhibition | 0.6667 | 0.4975 | 0.9949 | 0.6667 | 0.8308 | 0.0000 | 0.0000 | 0.0000 |
| Terminology preference | 0.3284 | 0.3284 | 0.0000 | 0.0000 | 0.3284 | 0.0000 | 0.0000 | 0.0000 |
| Default response language | 0.0000 | 0.2173 | 0.0000 | 0.0000 | 0.6373 | 0.0000 | 0.0000 | 0.0000 |
| Structured response default | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Answer-length control | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Local factual update | 0.0000 | 0.5441 | 0.6667 | 0.6667 | 0.6667 | 0.0000 | 0.0000 | 0.0000 |
| Selective factual withholding | 0.0000 | 0.2222 | 0.6667 | 1.0000 | 0.3333 | 0.0000 | 0.0000 | 0.0000 |
| Resistance to false suggestions | 0.1667 | 0.4993 | 0.4882 | 0.2910 | 0.2896 | 0.1354 | 0.1354 | 0.0000 |

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/family_radar.pdf}
\caption{Family reward profiles corresponding to Table 3. Each spoke is the mean of three selected episodes. Both panels use the same axes and 0--1 scale. Zero values overlap, as do GLM and DeepSeek on resistance to false suggestions.}
\label{fig:radar}
\end{figure}

6.1-terra and 6.1-sol at low and high reasoning achieve 9/9 alias successes. Medium reaches 8/9 and Astra 6/9. All three 6.1-sol settings succeed on all 112 remapping targets, against Astra's 99 and 6.1-terra's 94. Their remapping rewards differ because control retention differs. Medium has the highest lexical-inhibition reward at 0.9949, while Astra has the highest language-default reward at 0.6373.

6.1-terra, 6.1-sol at low reasoning and Astra each succeed on one of four terminology targets. The remaining configurations succeed on none. Selective withholding rises from 2/9 successes at low reasoning to 6/9 at medium and 9/9 at high. Astra reaches 3/9, while the remaining editors reach zero.

### Structured defaults and answer length

Every editor finishes with 0/192 structured-response successes, matching the pristine target under the required JSON convention.

Answer-length control also yields no positive reward. The pristine target succeeds on 2/9 items. 6.1-terra, 6.1-sol at medium reasoning, GLM and Nemotron retain 2/9. 6.1-sol at high reasoning finishes at 1/9, while low reasoning, Astra and DeepSeek finish at 0/9. Both the word band and required factual propositions must pass.

The low-reasoning 6.1-sol plant-growth answer becomes shorter after editing but remains unsuccessful. In the structured-response example, both checkpoints extract the correct name and return it as plain text. The required format remains absent.

### Target changes and control failures

The low-reasoning 6.1-sol saffron-as-cube edit changes the answers for faces, edges and vertices from 3/3/3 to 6/12/8. It retains 70 of 71 baseline-correct controls. On the failed control, an instruction to copy saffron exactly produces cube instead. The edit establishes the requested association but also applies it during literal copying.

## What the editing trajectories show

### Adapting edits to target feedback

The trajectories show editors adapting a small set of operations to the target's responses. They copy or blend embedding rows, rescale lexical rows and add measured residual directions to output rows. Astra's 273 successful write requests target the tied embedding/output matrix, and all thirty final patches modify only that matrix.

In the Canada factual-update episode, Astra adds a small multiple of a measured residual to the output row for the new answer's first token. Two of three terminal questions then produce Mark Carney, while the surname-only question returns Martin. All 67 scored controls remain correct. Low-reasoning 6.1-sol initially produces repeated Mark tokens, reduces the intervention and edits a surname-associated row after obtaining Mark Sharma. It also reaches two of three targets. 6.1-terra attempts a related edit but does not recover from overshooting and an invalidated activation handle. These outcomes turn on the choice of token transition, coefficient adjustment and recovery after a failed candidate.

Astra's lexical-inhibition trajectory shows repeated corrections to token-level effects. Suppressing doctor and teacher leaves capitalised forms intact. Suppressing case and plural variants produces fragments such as Doc and Teach. Astra then redirects the affected rows towards physician and instructor, correcting again when Physician becomes Phys. The final patch improves target success from one of two to two of two and retains all 66 baseline-correct controls.

### Practice performance and terminal behaviour

In Astra's structured-response episode, residual-based updates to the JSON-opening row initially cause repetition. Contrastive edits then produce the requested JSON on both practice names and preserve prose on a tested override. All 64 terminal targets still fail under a different extraction wording. Their names are correct, but the responses remain bare text. The edit fits the practice prompts without establishing the format under the terminal wording.

Astra's withholding trajectory uses vocabulary rows as temporary storage for measured vectors, preserving them across writes that invalidate activation handles. Further edits add contrasting France and copying contexts and a termination-related correction. The final patch succeeds on all three targets and retains all seventy scored controls. Additional probes produce a malformed answer to a broader Japan-capital question and still recover the reverse association. Zeroing the temporary storage rows leaves 25,600 coordinates changed relative to the pristine checkpoint.

### The marginal contribution of reasoning effort

Increasing 6.1-sol's reasoning effort changes individual outcomes more than its aggregate score. Mean reward rises from 0.4250 at low effort to 0.4601 at medium and 0.4551 at high. Medium improves six paired episode rewards, worsens six and ties eighteen. High improves six, worsens eight and ties sixteen. Reasoning-token use across the completed episodes rises from 13,631 to 48,456 and 174,770, respectively, approximately 3.6 and 12.8 times the low-effort total. The reward gain is modest and does not increase monotonically with effort.

Selective withholding accounts for the clearest improvement. Target successes rise from two of nine at low effort to six at medium and nine at high. Both higher-effort configurations lose the low-effort successes in default response language and terminology preference. All three solve every remapping target, and none earns positive reward on structured defaults or answer-length control. Excluding withholding, mean reward across the other nine families falls from 0.4475 at low effort to 0.4371 at medium and 0.3945 at high. Additional reasoning therefore improves one family substantially while leaving the broader pattern of failures in place.

### What appears to limit weight-editing skill

The trajectories show a recurring gap between finding an effective edit and controlling its scope. An alias edit changes copying, a new full name fails on a surname-only question, and a JSON convention succeeds on practice prompts but fails under another wording. The editor must choose both an intervention and the observations that reveal where it stops working.

A plausible explanation for the stronger models' performance is an effective policy for experimenting with weights. They select familiar operations, observe the effects and repair specific failures, as Astra does with tokenisation and the Canada edit. The reasoning-effort results suggest that longer deliberation alone does little to improve this policy across tasks. Withholding improves substantially, but much of the remaining performance is unchanged or worse. Better probe selection and interventions that transfer across prompts offer a more specific direction for improvement than increasing reasoning effort alone.

The environment can support reinforcement learning of this experimentation policy. The reward must make transfer and selectivity worth learning. A high pooled retention score already gives considerable credit to the saffron edit despite its copying failure. Varying prompt formulations and giving nearby and override failures explicit weight would train editors to distinguish a successful local adjustment from a change that meets the full objective. Such training could improve which operation the editor chooses, what it tests next and when it rejects a candidate.

## Conclusion

IWIBench makes gradient-free intraweight injection an empirical object of study. Defined tasks, constrained target access and paired evaluations allow us to measure what an edit achieves and audit the operations that produced it. The trajectories show differences in how models choose interventions, use feedback and recover from errors.

Its usefulness depends strongly on the task. Some editors change aliases and output labels successfully, yet none establishes the required structured-response or answer-length defaults. Intraweight injection is therefore a selective editing tool, not a general rewriter of model behaviour. The family profiles identify both tasks for which it works and objectives that remain difficult.

The tested frontier editors, 6.1-sol, 6.1-terra and Astra, also outperform the tested near-frontier peers by a wide margin. Their macro rewards range from 0.2928 to 0.4601, compared with 0.0135 to 0.0548 for GLM, DeepSeek and Nemotron. This separation, together with the differences in their editing trajectories, establishes a measurable distinction in weight-editing skill among the evaluated models.

\input{bibliography.tex}

\appendix

## Preservation controls

### Control groups

Each episode contains 64 general controls covering arithmetic, factual questions and exact copying.

Nearby controls test interference with related responses. Their specificity criterion parallels CounterFact+, where evaluation checks whether a factual edit changes related associations that should remain stable \cite{ref4}. Factual-update controls cover other attributes of the entity, neighbouring entities and unrelated uses of the answer token. Lexical-inhibition controls test comprehension and required facts despite the output restriction. Table 1 lists the controls for each family.

Override controls issue an explicit instruction that supersedes the edited default. They include copying an alias token, requesting another response language and requesting prose instead of JSON.

### Baseline-conditional retention

Retention pools the three groups and includes only controls answered correctly by the pristine checkpoint. Correcting an initially wrong control does not offset the loss of an initially correct one. The thirty selected episodes contain 2,288 baseline-correct controls.

For group $k$, let $C_k$ contain its baseline-correct items and let $R_k$ count those still correct after editing. Retention is $r = \sum_k R_k/\sum_k |C_k|$. Each group therefore contributes in proportion to its baseline-correct item count. Nearby spillover and override accuracy are also reported separately.

## Model versions and configurations

The target is `google/gemma-3-4b-it` at revision `093f9f388b31de276ce2de164bdc2081324b9767`, with native BF16 weights and text-only inputs. Terminal decoding is greedy with a 256-token generation cap. The evaluation uses family version 0.2.2 and grader version 0.2.3.

6.1-terra uses `gpt-6.1-terra`, temperature zero and no reasoning. 6.1-sol uses `gpt-6.1-sol` at low, medium and high reasoning effort. Astra uses `gpt-6-astra` at low reasoning effort. These configurations omit temperature and call the Responses API. GLM 5.3, DeepSeek V4.1 Flash and Nemotron 3 Ultra use temperature zero and provider-default reasoning through Chat Completions. All eight configurations cap editor responses at 4,096 tokens.

The third-party identifiers are `glm-5.3`, `deepseek-v4.1-flash-1017` and `nemotron-3-ultra-550b-a55b`. These configurations use Chat Completions through CheaperInference.

## Episode counts and resource use

Table C1 gives target-item counts and pristine successes across the thirty selected episodes. Table C2 reports target-access resource use.

**Table C1. Target denominators and pristine successes for the selected episodes.**

| Family | Pristine successes | Target items |
|---|---:|---:|
| Alias acquisition | 3 | 9 |
| Output-code remapping | 0 | 112 |
| Lexical inhibition | 1 | 6 |
| Terminology preference | 0 | 4 |
| Default response language | 0 | 9 |
| Structured response default | 0 | 192 |
| Answer-length control | 2 | 9 |
| Local factual update | 0 | 9 |
| Selective factual withholding | 0 | 9 |
| Resistance to false suggestions | 98 | 192 |
| Total | 104 | 551 |

**Table C2. Recorded target-access resource use across thirty episodes. These are observed totals, not maximum budgets.**

| Editor | Admitted tool calls | Generation calls | Inspection calls | Target forward tokens |
|---|---:|---:|---:|---:|
| 6.1-terra | 684 | 148 | 103 | 48,395 |
| 6.1-sol (low) | 627 | 170 | 86 | 35,488 |
| 6.1-sol (medium) | 958 | 219 | 100 | 80,089 |
| 6.1-sol (high) | 1,363 | 234 | 113 | 109,310 |
| Astra | 561 | 161 | 86 | 38,844 |
| GLM 5.3 | 84 | 17 | 9 | 5,013 |
| DeepSeek V4.1 Flash | 1,891 | 177 | 114 | 31,458 |
| Nemotron 3 Ultra | 152 | 28 | 5 | 1,910 |

Target forward tokens count generation and inspection. Provider-measured editor input and output tokens are recorded separately.

## Example identifiers

| Case | Editor | Trace ID |
|---|---|---|
| Saffron alias and copying failure | 6.1-sol (low) | `a46010909ab348c8babcdd1ab506d410` |
| Lexical inhibition | 6.1-terra | `b993d4156d6a4b04b3d915520302e9a5` |
| Resistance to false suggestions | 6.1-sol (low) | `216e227c0fde434e99ba82f7d0f8daed` |
| Structured response default | 6.1-sol (low) | `cd621cb820a14702845e3d439e618f13` |
| Answer-length control | 6.1-sol (low) | `3880934a78e147afb6c48bdf187bf77d` |
| Canada factual update | Astra | `85f4f0924ddd4f7286529ee4b36f2123` |
| Lexical-inhibition trajectory | Astra | `eaebc905eaf1494998edb06fc4a897ea` |
| Structured-response trajectory | Astra | `83f7d72c0fca4417b21ce07d1cb0e968` |
| Withholding trajectory | Astra | `e512adcad9054361a5dcf52a6116ed20` |

