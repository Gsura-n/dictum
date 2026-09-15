# Architecture

## The problem, split into four

"Replicate Wispr Flow locally" sounds like a speech recognition problem. It is mostly not. It is four separate problems that have to feel like one:

1. Capture: a global hotkey that starts and stops the microphone from any app.
2. Speech to text: fast enough that the result appears about as soon as you release the key.
3. Refinement: an LLM pass that removes filler, applies self-corrections, punctuates, and adapts tone to the target.
4. Injection: putting the text into whatever field has focus, in any app, as if you typed it.

Each has a different failure mode, a different set of dependencies, and a different platform surface. So each is its own stage with its own interface, and `Pipeline` is the only place they touch.

```
dictum/
  capture/   AudioCapture   start() / stop() -> AudioClip
  stt/       STTEngine      transcribe(AudioClip) -> Transcript      + registry
  refine/    Refiner        refine(Transcript, Mode) -> Refined
  inject/    Injector       inject(str)
  pipeline.py               Pipeline.from_config(cfg).process(clip)
  config.py                 YAML with local overrides; modes and dictionary live here
  types.py                  AudioClip, Transcript, Refined
```

Interfaces are `typing.Protocol`, not abstract base classes. An adapter is any class with the right methods; nothing to inherit, nothing to register except one line in the STT registry. Optional dependencies are imported lazily inside adapters so `pip install dictum` never pulls in three model runtimes you do not use.

## Decisions and why

**Pluggable STT with two adapters from day one.** Model choice in this space changes every few months. The registry means a new engine is one file. Shipping two adapters also makes the benchmark meaningful: the point is to pick based on measured latency and WER on the user's own voice, not on a leaderboard.

**Parakeet TDT as default, Whisper as the known quantity.** Parakeet has a lower real-time factor at equal or better English accuracy. Whisper is multilingual and familiar. Both run through MLX so they use the Apple Silicon GPU without a CUDA dependency.

**Modes are configuration, not code.** A mode is a name, a system prompt, a few worked examples and optionally a model override, all in YAML. Dictation, command and academic are just three entries. Users add their own without touching Python. In phase 4 a mode can also be selected automatically by the focused application.

**Few-shot examples over longer rule lists.** The refiner runs on a 3B model to stay inside the latency budget, and small models follow two worked examples far more reliably than ten bullet points. Examples live next to the prompt in the mode definition so they can be tuned with `dictum refine` without touching code.

**Per-mode model selection.** Dictation wants speed (llama3.2:3b). Command mode wants a code model (qwen2.5-coder:7b). Academic mode can afford a larger general model because nobody dictates a paragraph of a paper and expects it in half a second. Routing by mode keeps the common path fast without capping quality on the rare paths.

**Defensive output cleanup.** Small models occasionally wrap output in quotes or code fences, and reasoning models emit think blocks. The refiner strips all three rather than trusting the prompt.

**Refinement is a stage, not a feature of STT.** Keeping it separate means it can be turned off (passthrough), swapped (Ollama today, MLX-native later), and measured on its own. The latency budget is the whole point: if STT plus refine exceeds about 1.5 seconds, people stop using push-to-talk tools.

**Injection via clipboard paste, not per-character keystrokes.** Simulated typing is slow and breaks on non-ASCII. Clipboard plus Cmd+V with clipboard restore is what mature tools do. It needs Accessibility permission; the stdout and clipboard injectors exist so the pipeline is usable before that permission is granted.

**A raw Quartz event tap for the hotkey, not pynput.** pynput's macOS backend calls text input APIs off the main thread, which crashes on recent macOS. A listen-only `CGEventTap` on the main run loop is short, reads device-dependent modifier flags so Right Option and Left Option are different keys, and gives full control over edge cases: another key pressed while holding cancels the dictation, so Option shortcuts keep working.

**Three threads, each with one job.** The main thread runs the Quartz run loop (macOS requires it). A recorder thread opens and closes the mic, because opening a Bluetooth input can take hundreds of milliseconds and the tap callback must return instantly or macOS disables it. A processor thread owns the models, since MLX state is per thread. Queues connect them, so dictating twice quickly just queues the second clip.

**Clipboard restore that respects the user.** The injector snapshots every type on every pasteboard item (text, images, files), pastes, waits for the target app to read, and restores only if the clipboard has not changed since. A screenshot you copied before dictating is still there after.

**Degrade, never lose words.** If Ollama is down or the model is missing, the raw transcript is pasted and a warning printed. If STT returns nothing, the LLM is never called, because a small model given an empty prompt will invent a plausible sentence (this happened during development).

**Capture at the device's native rate.** Bluetooth headsets such as AirPods run their mic at 24 kHz and can return pure silence when opened at 16 kHz. Capture opens the device at whatever it reports and resamples with soxr.

**Measure before tuning.** `dictum eval` runs the refine stage alone against cases scored by deterministic checks (required and forbidden phrases, regexes, length ratios) rather than exact matches, reported by difficulty with latency percentiles. Guard fallbacks are counted separately so a guard rescuing a bad output cannot hide in the pass rate. Few-shot examples are kept disjoint from eval cases.

**The model proposes, code disposes.** Anything that must be right every time lives in code: dictionary respellings, the output length guard that replaces a hallucinated poem with the user's own words, single-line enforcement for commands. The LLM handles only what genuinely needs judgment.

**Mode is resolved at key press, from the app in front.** The frontmost app at press time is where the text will land, so that is when `auto` is resolved. A second key always forces plain dictation, because a terminal is also where you write prompts and commit messages.

**Command mode never executes.** It produces a proposed command. Running it is a deliberate, separate action by the user.

## Latency budget (target on M4, 16 GB)

| stage | target |
|---|---|
| STT (5 s clip) | < 0.5 s |
| refine (3B model, short text) | < 0.8 s |
| inject | < 0.1 s |
| total after key release | < 1.5 s |

`dictum listen` prints all of these on every utterance so regressions are visible immediately.

## Roadmap

- Phase 1: capture, STT, benchmark, terminal output (done)
- Phase 2: Ollama refiner, modes with examples and per-mode models, dictionary (done)
- Phase 3: global hold-to-talk hotkey, paste into focused app with clipboard restore, permission checks (done)
- Phase 4: app-aware mode selection (started: frontmost app picks the mode), self-improving dictionary from user corrections, streaming partials, small overlay
