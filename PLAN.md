# PLAN.md

Scope grew past the original <2hr lightweight tokeniser exercise into a full
"build an LLM from first principles" learning project. Broken down into
tasks in TASK.md from this point on.

## Approach
First-principles implementation, all the way down. No ML/tokeniser/tensor
libraries (no PyTorch, tiktoken, SentencePiece, HuggingFace, ML.NET,
Math.NET, System.Numerics.Tensors, etc.). Tokenisation, tensor math,
autodiff, attention, and the training loop are all written by hand in plain
C# so every gradient and every mechanism is visible and understood, not a
black box behind a library call. This is a deliberate, explicit trade-off:
slower to reach a working model, but the whole point of the project is
understanding the mechanics, not shipping a model quickly.

## Goal
Learn how LLMs work by building one end-to-end: tokeniser → tensors/autodiff
→ embeddings → attention/transformer → training loop → text generation.

## Stages

1. **Tokeniser (done)** — byte-level BPE: train a vocabulary from one or
   more text files or a directory of them, encode/decode, save/load. See
   `src/Tokeniser/`.
2. **Tensor + autodiff engine** — an N-dimensional array type with the core
   ops a transformer needs (add, multiply, matmul, transpose, softmax,
   etc.), plus reverse-mode automatic differentiation (a computation graph
   and a backward pass) so gradients don't have to be derived and coded by
   hand for every operation. This is the foundation everything else sits on.
3. **Embeddings** — a learned token-id → vector lookup table, plus
   positional encoding (attention has no inherent sense of sequence order).
4. **Attention + transformer block** — scaled dot-product attention,
   multi-head attention, feed-forward layers, layernorm, residual
   connections — the actual transformer architecture.
5. **Model assembly** — stack transformer blocks into a decoder-only
   (GPT-style) model with an output projection back to vocabulary logits.
6. **Training loop** — batching token sequences from the tokeniser's output,
   next-token-prediction cross-entropy loss, an optimizer (SGD first, then
   Adam/AdamW), backprop via the autodiff engine, checkpointing.
7. **Generation** — greedy, temperature, and top-k/top-p sampling to
   actually produce text from a trained model.

## Notes / future ideas (not in scope yet)
- Regex-based pre-tokenisation (GPT-2 style splitting on whitespace/punctuation
  before BPE) — the current whole-text byte stream approach works; revisit
  only if it becomes a real limitation.
- GPU/parallel execution, mixed precision, distributed training — CPU-only,
  single-machine scope for now. This is a learning project, not a
  performance target.
