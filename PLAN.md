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

**All 7 stages are done (TASK-001 through TASK-013 in TASK.md).** The
project can train a small GPT-style model from scratch on a text corpus and
generate text from it, entirely from first principles. What's left is
tuning/scaling up an actual training run and addressing the deferred items
below, not new architecture.

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
- See `src/Common/MappedArray.cs` for the established pattern (shared by
  the tokeniser and the tensor engine)
  (memory-mapped scratch file wrapper) to reuse or adapt.

## Stages

1. **Tokeniser (done)** — byte-level BPE: train a vocabulary from one or
   more text files or a directory of them, encode/decode, save/load. See
   `src/Tokeniser/`.
2. **Tensor + autodiff engine (done)** — an N-dimensional array type with
   the core ops a transformer needs (add, multiply, matmul, transpose,
   softmax, etc.), plus reverse-mode automatic differentiation (a
   computation graph and a backward pass) so gradients don't have to be
   derived and coded by hand for every operation. See `src/Tensor/`.
3. **Embeddings (done)** — a learned token-id → vector lookup table, plus
   positional encoding (attention has no inherent sense of sequence order).
   See `src/Model/{TokenEmbedding,PositionalEmbedding}.cs`.
4. **Attention + transformer block (done)** — scaled dot-product attention,
   multi-head attention, feed-forward layers, layernorm, residual
   connections. See `src/Model/{ScaledDotProductAttention,
   MultiHeadAttention,FeedForward,LayerNorm,TransformerBlock}.cs`.
5. **Model assembly (done)** — stack transformer blocks into a decoder-only
   (GPT-style) model with a weight-tied output projection back to
   vocabulary logits. See `src/Model/GptModel.cs`.
6. **Training loop (done)** — batching token sequences from the tokeniser's
   output, next-token-prediction cross-entropy loss, an optimizer (SGD and
   AdamW), backprop via the autodiff engine, checkpointing. See
   `src/Training/`.
7. **Generation (done)** — greedy, temperature, and top-k/top-p sampling to
   actually produce text from a trained model. See `src/Generation/`.

## Known limitations / deferred (not bugs - documented trade-offs)
Flagged individually in TASK.md as each was made; collected here for a
single view of what a real training run might need to revisit:

- **Bulk-encoding a large corpus is slow.** `BpeTokeniser.Encode` uses the
  simple merge-scan approach tuned for short/moderate text (TASK-010),
  not the efficient training-time algorithm `Train` uses. Fine for a
  prompt; would need work for encoding a full large corpus into training
  data.
- **No KV-cache during generation** (TASK-013): every step recomputes a
  full forward pass over the whole context. Simple and correct, not fast.
- **Checkpoints and AdamW's moment estimates are plain heap allocations**
  (TASK-011/012), not disk-backed like the tokeniser's/tensor engine's
  large structures - nothing built so far has been large enough to need it.
- **Two places don't use the max-subtraction softmax stability trick**:
  `Variable.Softmax` (TASK-004, used in attention/training - would need a
  max-along-axis Tensor reduction that doesn't exist yet) and
  `CrossEntropyLoss`'s `log(sum(exp(x)))` term (TASK-011). Fine at the
  scales tested so far; `Generation.TokenSampler`'s softmax *does* use the
  trick, since it has no backward pass to complicate and temperature
  scaling can push values to real extremes.
- Regex-based pre-tokenisation (GPT-2 style splitting on whitespace/
  punctuation before BPE) — the current whole-text byte stream approach
  works; revisit only if it becomes a real limitation.
- GPU/parallel execution, mixed precision, distributed training — CPU-only,
  single-machine scope. This is a learning project, not a performance
  target.
