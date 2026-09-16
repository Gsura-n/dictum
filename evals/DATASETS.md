# Eval datasets and attribution

| suite | source | authored by | license | split used for tuning | split used for reporting |
|---|---|---|---|---|---|
| `disflqa` | [Disfl-QA](https://github.com/google-research-datasets/Disfl-QA), Gupta et al. 2021 | human annotators | CC BY 4.0 | 200 sampled from `dev` | 500 sampled from `test` |
| `disflspeech` | [DisfluencySpeech](https://huggingface.co/datasets/amaai-lab/DisfluencySpeech), AMAAI Lab 2024 (text from Switchboard) | human annotators | Apache 2.0 | 250 from `validation` | 250 from `test` |
| (training only) `swda` | [Switchboard Dialog Act Corpus](https://github.com/cgpotts/swda), Jurafsky et al. 1997 | human annotators (disfluency markup) | GPL-2.0 distribution | not used | not used |
| `nl2bash` | [NL2Bash](https://github.com/TellinaTool/nl2bash), Lin et al. 2018 | Bash programmers | MIT (data) | 100 from a seeded split of executable cases | 300 from the same |
| `targeted` | `evals/cases/*.yaml` | written for this project with an AI assistant | MIT | all | not used for headline numbers |

Rules:
- Prompts, examples and guards may be tuned against `dev` only. `test` is run to report numbers, and its failures are not read while tuning.
- Samples are drawn with a fixed seed so every run and every model sees the same cases.
- The Disfl-QA `train` split (7,182) is used only by `scripts/finetune/prepare_data.py`. Items whose text also appears in dev or test are dropped (29 of them). A 300-row validation set for training is carved from train, never from dev.
- NL2Bash rows outside the seeded dev and test slices are reserved as train and not used yet.
- Command cases are also scored by execution (`--exec`): both commands run in a no-network Docker sandbox on identical fixture files, and a case counts only if the reference itself runs. NL2Bash is scored with a Linux platform hint because the sandbox is Linux.

Known limitations:
- Disfl-QA disfluencies were typed by annotators, not transcribed from audio, and every item is a question. It is the best openly licensed human source of self-corrections, and a strong test of "clean up the question, do not answer it", but it is not spontaneous speech.
- NL2Bash targets Linux (GNU) tools. Items explicitly marked GNU-specific are dropped; others may still use GNU-only flags, so command scores against a macOS prompt are a lower bound.
- NL2Bash descriptions are written, not spoken.

Attribution: Disfl-QA © Google LLC, licensed CC BY 4.0. NL2Bash © the NL2Bash authors, MIT. Neither is redistributed in this repository; `scripts/fetch_datasets.py` downloads them.

- NL2Bash is filtered to cases whose reference command actually runs in the sandbox fixture (`dictum prep-nl2bash`). About a quarter do; the rest reference files, users or hosts that do not exist. This biases the suite toward file and text commands, which is also what people dictate.
- DisfluencySpeech pairs are `transcript_a` (non-speech events removed) to `transcript_c` (fillers, discourse markers, editing terms and false starts removed). Annotation marks are stripped, and targets get deterministic sentence casing and a final period. Only text columns are downloaded; the audio is not. Its train split is mixed with Disfl-QA train for fine-tuning, with dev/test overlaps removed.
- SwDA pairs are derived by rule from Meteer-style markup: fillers {F}, discourse markers {D} and editing terms {E} removed; in [ X + Y ] repairs only Y is kept; conjunctions {C} and asides {A} kept; non-speech and partial words removed. Consecutive units of one speaker turn are joined (up to 3). Because SwDA and DisfluencySpeech share Switchboard as a source, any SwDA turn sharing an 8-word sequence with a DisfluencySpeech dev/test sentence is dropped (453 turns).
- Training mix is capped per source (default Disfl-QA 4,000, DisfluencySpeech 4,500, SwDA 16,000, plus 20% identity rows) so no single source dominates.
