# Dictum

Local-first voice dictation for macOS. Hold a key, talk, get clean text in whatever app you are using. Nothing leaves your machine.

Think of it as an open source take on tools like Wispr Flow, with two differences: it runs fully offline on Apple Silicon, and it has explicit modes (plain dictation, shell command, academic prose) instead of one generic cleanup.

## Status

Phase 3 of 4. What works today:

- Hold Right Option anywhere on your Mac, talk, release: cleaned text is pasted into whatever app has focus. Right Command does the same in command mode. Your clipboard is restored afterwards
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
