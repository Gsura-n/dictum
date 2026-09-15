# Eval datasets and attribution

| suite | source | authored by | license | split used for tuning | split used for reporting |
|---|---|---|---|---|---|
| `disflqa` | [Disfl-QA](https://github.com/google-research-datasets/Disfl-QA), Gupta et al. 2021 | human annotators | CC BY 4.0 | 200 sampled from `dev` | 500 sampled from `test` |
| `nl2bash` | [NL2Bash](https://github.com/TellinaTool/nl2bash), Lin et al. 2018 | Bash programmers | MIT (data) | 100 from a seeded split | 300 from a seeded split |
| `targeted` | `evals/cases/*.yaml` | written for this project with an AI assistant | MIT | all | not used for headline numbers |

Rules:
- Prompts, examples and guards may be tuned against `dev` only. `test` is run to report numbers, and its failures are not read while tuning.
- Samples are drawn with a fixed seed so every run and every model sees the same cases.
- The `train` portions (Disfl-QA 7,182; NL2Bash ~8,000) are untouched and reserved for possible fine-tuning. They never overlap with `dev` or `test`.

Known limitations:
- Disfl-QA disfluencies were typed by annotators, not transcribed from audio, and every item is a question. It is the best openly licensed human source of self-corrections, and a strong test of "clean up the question, do not answer it", but it is not spontaneous speech.
- NL2Bash targets Linux (GNU) tools. Items explicitly marked GNU-specific are dropped; others may still use GNU-only flags, so command scores against a macOS prompt are a lower bound.
- NL2Bash descriptions are written, not spoken.

Attribution: Disfl-QA © Google LLC, licensed CC BY 4.0. NL2Bash © the NL2Bash authors, MIT. Neither is redistributed in this repository; `scripts/fetch_datasets.py` downloads them.
