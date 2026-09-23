# Reported results

The eval result files behind every number in the top-level README. `evals/results/`
is git-ignored scratch space; a run is copied here only when it is quoted.

| file | suite | run on | adapter |
|---|---|---|---|
| 20260916-231606-disflqa-test.json | Disfl-QA test (500) | 2026-09-16 | adapters/dictation-v2 |
| 20260916-223253-disflspeech-test.json | DisfluencySpeech test, transcripts (250) | 2026-09-16 | adapters/dictation-v2 |
| 20260916-215554-disflspeech-test-audio.json | DisfluencySpeech test, audio through Parakeet (250) | 2026-09-16 | adapters/dictation-v2 |

All three ran on an M4 Mac mini with `mlx-community/Llama-3.2-3B-Instruct-4bit`
plus the adapter, mode `dictation_ft`. Each file has every case: input,
reference, output, pass/fail, failure reason and latency.

The runs predate the edit mode and chunking changes in 6574194. They will be
rerun on current code before the next README update.
