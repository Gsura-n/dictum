# Dictum

Local-first voice dictation for macOS. Hold a key, talk, get clean text in whatever app you are using. Nothing leaves your machine.

Think of it as an open source take on tools like Wispr Flow, with two differences: it runs fully offline on Apple Silicon, and it has explicit modes (plain dictation, shell command, academic prose) instead of one generic cleanup.

## Status

Phase 1 of 4. What works today:

- Push-to-talk recording from the terminal
- Speech to text with a pluggable engine layer: NVIDIA Parakeet TDT (default) or Whisper, both via MLX
- A benchmark command that compares engines on latency and word error rate on your own voice

Coming next: LLM cleanup through Ollama (phase 2), typing into the focused app with a global hotkey (phase 3), then modes, personal dictionary and an overlay (phase 4).

## Quick start (Apple Silicon)

```bash
git clone <repo> dictum && cd dictum
./scripts/setup.sh
source .venv/bin/activate
dictum devices     # confirm your microphone shows up
dictum listen      # Enter to start, Enter to stop; first run downloads the model
```

Try the other engine:

```bash
dictum listen --engine whisper
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
