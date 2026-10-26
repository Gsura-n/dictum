# Reported results

The eval result files behind every number in the top-level README. `evals/results/`
is git-ignored scratch space; a run is copied here only when it is quoted.

| file | suite | run on | adapter |
|---|---|---|---|
| 20260916-231606-disflqa-test.json | Disfl-QA test (500) | 2026-09-16 | adapters/dictation-v2 |
| 20260916-223253-disflspeech-test.json | DisfluencySpeech test, transcripts (250) | 2026-09-16 | adapters/dictation-v2 |
| 20260916-215554-disflspeech-test-audio.json | DisfluencySpeech test, audio through Parakeet (250) | 2026-09-16 | adapters/dictation-v2 |
| 20261018-230429-disflqa-test.json | Disfl-QA test (500), rerun | 2026-10-18, commit f7568f5 | adapters/dictation-v2 |
| 20261018-230821-disflspeech-test.json | DisfluencySpeech test, transcripts (250), rerun | 2026-10-18, commit f7568f5 | adapters/dictation-v2 |
| 20261018-231256-disflspeech-test-audio.json | DisfluencySpeech test, audio through Parakeet (250), rerun | 2026-10-18, commit f7568f5 | adapters/dictation-v2 |
| baseline-20260923-150648.json | end-to-end latency and memory, 40 x ~10 s DisfluencySpeech dev audio | 2026-09-23, commit 6d2cb1f | adapters/dictation-v2 |

The six test-split files ran on an M4 Mac mini with `mlx-community/Llama-3.2-3B-Instruct-4bit`
plus the adapter, mode `dictation_ft`. Each file has every case: input,
reference, output, pass/fail, failure reason and latency.

The 2026-10-18 files were produced with `dictum eval --suite <disflqa|disflspeech> --split test --backend mlx
--as-mode dictation_ft`, plus `--audio real` for the audio file. The 2026-09-16 files predate the edit mode and
chunking changes in 6574194 and the normalize fix in fb24ad0; both sets are kept so the change is visible.
