# PLAN.md

Lightweight project (planning process waived per CLAUDE.md rules for <2hr projects).

## Approach
First-principles implementation. No ML/tokeniser libraries (no PyTorch,
tiktoken, SentencePiece, HuggingFace, etc.) — the BPE algorithm itself
(pair-frequency counting, merge learning, encode/decode) is written by hand
in plain C# so the mechanics are fully visible.

## Goal
Learn how LLMs work, starting with tokenisation. Build a Byte-Pair Encoding (BPE)
tokeniser in C#/.NET that:

- Takes one or more plain text files as input.
- Trains a BPE vocabulary from their contents (learns merges from byte/character
  pair frequency, like GPT-style tokenisers).
- Encodes new text into token IDs using the learned vocabulary.
- Decodes token IDs back into text.
- Can save/load the trained vocabulary + merge rules to/from disk.

## Notes / future ideas (not in scope yet)
- Regex-based pre-tokenisation (GPT-2 style splitting on whitespace/punctuation
  before BPE) — start simpler (whole-text byte stream) first.
- Embeddings, attention, etc. — later learning stages, not part of this project.
