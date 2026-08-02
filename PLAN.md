# PLAN.md

Scope grew past the original <2hr lightweight tokeniser exercise into a full
"build an LLM from first principles" learning project. Broken down into
tasks in TASK.md from this point on.

## Approach
First-principles implementation, all the way down, **by default**. No
ML/tokeniser/tensor libraries (no PyTorch, tiktoken, SentencePiece,
HuggingFace, ML.NET, Math.NET, System.Numerics.Tensors, etc.). Tokenisation,
tensor math, autodiff, attention, and the training loop are all written by
hand in plain C# so every gradient and every mechanism is visible and
understood, not a black box behind a library call. This is a deliberate,
explicit trade-off: slower to reach a working model, but the whole point of
the project is understanding the mechanics, not shipping a model quickly.

**One explicit, opt-in exception (stage 9 / TASK-015):** an optional
library-backed fast path for the tensor engine's hot ops, off by default,
enabled only via an explicit flag. The hand-written scalar implementation
stays the real, default, always-correct reference; the fast path is a
strictly optional accelerant for anyone who wants to actually run a
larger training job on this hardware, not a replacement for the
first-principles version or a quiet erosion of the "no libraries" rule.
See stage 9 below for why this is scoped as narrowly as it is.

## Goal
Learn how LLMs work by building one end-to-end: tokeniser → tensors/autodiff
→ embeddings → attention/transformer → training loop → text generation.

**The original 7 stages are done (TASK-001 through TASK-013 in TASK.md).**
The project can train a small GPT-style model from scratch on a text corpus
and generate text from it, entirely from first principles. Stage 8
(interactive CLI) is a scope addition on top of that original plan — see
below.

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
8. **Interactive CLI ("chat")** — a console app that loads a saved
   `ModelCheckpoint` + tokeniser vocab and lets the user converse with the
   model turn-by-turn interactively, instead of only being usable via a
   one-off code snippet (README's "Full workflow" section). Added after
   the original 7-stage plan, at the user's request.

   **Honest expectation-setting, not a limitation to fix later:** this
   model is trained by plain next-token prediction on raw text, not on
   chat-formatted/instruction data with turn markers or roles. A CLI loop
   makes it *usable* conversationally (keeps a running token context across
   turns, doesn't require re-invoking a program per message), but it won't
   make the model behave like an instruction-following assistant unless the
   corpus it was trained on was itself shaped like dialogue. Worth being
   upfront about in the CLI's own `--help`/README text, not just here.
9. **Optional optimised math backend (`--optimised`)** — an opt-in fast
   path for `Tensor`'s hot ops (matmul above all; possibly elementwise ops
   too) backed by Math.NET Numerics and/or `System.Numerics.Tensors`,
   selected by an explicit flag (default: off, i.e. the existing
   hand-written scalar path). Added after the original 7-stage plan, at
   the user's request, alongside stage 8.

   **Why this is scoped narrowly, and the open design questions a real
   implementation needs to resolve:**
   - The standing memory-discipline constraint (PLAN.md, above) still
     applies regardless of backend. `Tensor` already separates storage
     (`IFloatBuffer`: heap vs disk-backed `MappedArray<T>`) from compute;
     a library-backed compute path most likely only benefits heap-backed
     tensors, since Math.NET/`System.Numerics.Tensors` types expect
     contiguous managed memory, not our disk-backed buffer abstraction.
     Whether/how a large disk-backed tensor could still get *any* benefit
     (e.g. computing over bounded chunks) is an open question, not a given.
   - `System.Numerics.Tensors` ships in the .NET BCL (no NuGet needed);
     Math.NET Numerics is a real external package. "No libraries" was
     never really about install friction - it's about not hiding the
     mechanics behind someone else's code. An opt-in flag keeps that
     distinction honest: the default experience is still 100%
     first-principles; `--optimised` is a clearly-labelled, deliberate
     exception someone reaches for, not a silent swap.
   - Needs a way to thread the choice from a CLI flag down to `Tensor`
     construction/ops without polluting every call site - likely a
     process-wide switch set once at startup (e.g. a static
     `Tensor.Backend` or similar), not a parameter added to every op.
   - Correctness: the optimised path must be verified to produce results
     equivalent (within float tolerance) to the existing scalar path -
     the natural way to prove this is running the *same* test suite
     (including the finite-difference gradient checks) against both
     backends, not writing a separate, smaller test suite for the fast
     path.
   - Priority: matmul is the dominant cost in a transformer forward/backward
     pass (O(n³) vs elementwise ops' O(n)), so it's the one actually worth
     optimising first; elementwise ops are a "nice to have" stretch goal on
     top, not required for the flag to be worth having at all.

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
