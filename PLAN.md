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

## Memory constraint (standing, applies to every stage)
This machine runs with little RAM to spare (earlier tokeniser work OOM-killed
the box more than once — see commit history). Every stage that handles data
or state scaling with corpus/model size must treat disk as scratch space
rather than assuming it fits in RAM:

- Prefer memory-mapped or streamed disk-backed storage over large in-memory
  arrays/collections wherever a structure scales with corpus size, dataset
  size, or model size (token streams, activations/gradients if they get
  large, checkpoints, batched training data).
- Minimise heap (anonymous) memory specifically: the OS can reclaim
  file-backed (disk-mapped) pages directly under pressure, but anonymous
  heap memory can only be reclaimed via swap — which is what actually causes
  an OOM kill on this machine. When choosing between "bigger disk-backed
  structure" and "bigger heap-allocated structure," prefer the former.
- Before adding a new large in-memory structure (e.g. a batch of
  activations, a full-corpus token buffer, an optimizer's moment estimates),
  think through its memory scaling and prefer a bounded/streamed/disk-backed
  approach over one that scales unbounded with input size.
- This applies most directly to TASK-010 (batching) and TASK-012 (training
  loop), where dataset size and model state are the largest consumers, but
  keep it in mind for the tensor engine (TASK-003/004) too, since tensors
  are the thing everything else scales through.
- See `src/Tokeniser/MappedInt32Array.cs` for the established pattern
  (memory-mapped scratch file wrapper) to reuse or adapt.

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
