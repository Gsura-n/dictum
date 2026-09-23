# Dictum

Local-first voice dictation for macOS. Hold a key, talk, get clean text in whatever app you are using. Nothing leaves your machine.

Think of it as an open source take on tools like Wispr Flow, with two differences: it runs fully offline on Apple Silicon, and it has explicit modes (plain dictation, shell command, academic prose) instead of one generic cleanup.

## Status

Phase 3 of 4. What works today:

- Hold Right Option anywhere on your Mac, talk, release: cleaned text is pasted into whatever app has focus. Right Command always uses plain dictation. Right Control adds a copy-edit pass that breaks long speech into sentences and paragraphs, for when you are writing rather than noting. Your clipboard is restored afterwards
- A terminal push-to-talk mode (`dictum listen`) for testing each stage
- Speech to text with a pluggable engine layer: NVIDIA Parakeet TDT (default) or Whisper, both via MLX
- LLM cleanup through Ollama, with modes: dictation (filler removal, self-corrections, punctuation), command (speech to one shell command, never executed), academic (APA style prose). Each mode has its own prompt, few-shot examples and optionally its own model
- A personal dictionary that keeps your names and jargon spelled right
- A benchmark command that compares STT engines on latency and word error rate on your own voice

Coming next (phase 4): app-aware mode selection, a self-improving dictionary learned from your corrections, streaming partial results and a small overlay.

## Quick start (Apple Silicon)

```bash
git clone <repo> dictum && cd dictum
./scripts/setup.sh
source .venv/bin/activate
dictum check       # permissions, microphone, Ollama models
dictum run         # hold Right Option anywhere and talk
```

`dictum run` needs two macOS permissions for your terminal app, both under System Settings > Privacy & Security: **Input Monitoring** (to see the hotkey) and **Accessibility** (to paste). Grant them, then quit and reopen the terminal. `dictum check` tells you what is missing.

Refinement needs [Ollama](https://ollama.com) running with a model pulled (default `llama3.2:3b`; command mode uses `qwen2.5-coder:7b`, academic uses `llama3.1:8b`, all configurable):

```bash
ollama pull llama3.2:3b
dictum listen                       # dictation mode
dictum listen --mode command        # speech -> shell command, printed not run
dictum refine "um so I I think we should uh ship it friday no monday" --all   # iterate on prompts without the mic
dictum listen --refine passthrough  # raw STT only
dictum listen --engine whisper      # the other STT engine
```

Record a few clips and compare engines on them:

```bash
dictum listen --save benchmarks/samples
# type what you said into benchmarks/samples/clip_001.txt etc. for WER
dictum bench
```

## How good is it?

Evals run the refine stage alone (STT is measured separately by `dictum bench`) and report pass rate with a 95% confidence interval.

Headline numbers come from human-authored public datasets, split into a dev set that prompts are tuned against and a test set that is only run to report. See [evals/DATASETS.md](evals/DATASETS.md).

```bash
python scripts/fetch_datasets.py
dictum eval --suite disflqa --split test     # 500 human-written disfluent questions
dictum eval --suite nl2bash --split test     # 300 programmer-written command descriptions
dictum eval --suite targeted                 # 30 hand-written regression cases
```

Results on the held-out test splits for the default dictation mode (`dictation_ft`: Llama 3.2 3B 4-bit plus the fine-tuned LoRA adapter), measured on an M4 Mac mini on 2026-09-16, before the chunking change in 6574194 (a rerun on current code is pending). Each number links to the saved result file with every case in it.

| suite | cases | pass rate (95% CI) | refine p50 / p95 |
|---|---|---|---|
| [Disfl-QA test](evals/reported/20260916-231606-disflqa-test.json) | 500 | 73% (69 to 77) | 0.61 s / 0.85 s |
| [DisfluencySpeech test, transcripts](evals/reported/20260916-223253-disflspeech-test.json) | 250 | 92% (88 to 95) | 0.90 s / 1.21 s |
| [DisfluencySpeech test, from audio](evals/reported/20260916-215554-disflspeech-test-audio.json) | 250 | 71% (65 to 76) | 0.88 s / 1.19 s |

The audio row runs the recordings through Parakeet first, so speech recognition errors count against it. Latency is the refine stage only; end-to-end latency on the minimum supported hardware is not measured yet. The test splits were not used for tuning.

Early results on the 30 hand-written targeted cases (written with an AI assistant, used for regression, not for headline claims):

| mode | model | before tuning | after | p50 refine |
|---|---|---|---|---|
| dictation | llama3.1:8b | 59% | 94% | 1.07 s |
| dictation | llama3.2:3b | 53% | 76% | 0.40 s |
| command | qwen2.5-coder:7b | 92% | 100% | 0.57 s |

What moved those numbers was mostly not the model: delimiting the transcript so it is treated as data (this fixed prompt injection), deterministic output guards, fixing dictionary words in code before the LLM runs, and few-shot examples kept separate from eval cases.

## Personalization (optional, local)

The default dictation mode (`dictation_ft`) uses a LoRA adapter fine-tuned on Llama 3.2 3B. **The adapter is not published yet.** Until it is, a fresh clone falls back to the base 3B model, which is much weaker at cleanup (4% on Disfl-QA dev against 76% with the adapter). In the meantime, train it with `scripts/finetune/train.sh`, or set `apps.default: dictation` and `right_command: dictation` in `config/local.yaml` to use the prompted Ollama model instead.

Once you have an adapter, you can teach it your own vocabulary and phrasing:

```bash
dictum history                  # after setting history.enabled: true in config/local.yaml
dictum correct --text "what it should have been"    # or --ok when the output was right
dictum personalize              # after 50+ corrections: trains on top of the current adapter
dictum personalize --reset      # back to the base adapter
```

Personalization continues training from the current adapter on your corrections mixed with general examples, scores old vs new on a held-out slice of your own corrections, and only switches if the new one is at least as good. History and adapters live in `~/.dictum/` and never leave the machine.

## Configuration

`config/default.yaml` holds everything: audio device, engine and model names, refinement backend, injection backend, modes and your personal dictionary. Copy it to `config/local.yaml` for overrides; that file is git-ignored.

## Architecture

Four stages, each behind a small interface, wired together in one place:

```
Capture  ->  STT  ->  Refine  ->  Inject
 (mic)     (Parakeet   (LLM per    (stdout |
            | Whisper)   mode)     clipboard |
                                    focused app)
```

See `ARCHITECTURE.md` for the reasoning behind each choice.

## Why Parakeet by default

Parakeet TDT 0.6B v2 sits at the top of the Open ASR leaderboard for English with a much lower real-time factor than Whisper large-v3. On an M4 that means transcription finishes well under a second for typical dictation. Whisper is kept as a second adapter because it covers more languages and is the model most people already know, and because a pluggable layer is only credible if there are at least two things plugged in. Run `dictum bench` to see the numbers on your own machine and voice rather than trusting a leaderboard.

## License

MIT
