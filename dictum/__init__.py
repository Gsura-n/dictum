"""Dictum: local-first voice dictation pipeline.

Capture -> STT -> Refine -> Inject. Each stage is a small interface with
swappable implementations; see ARCHITECTURE.md.
"""
__version__ = "0.1.0"
