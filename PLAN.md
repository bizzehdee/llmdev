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

**One explicit, opt-in exception, scoped to exactly two named libraries
(stage 9 / TASK-015):** Math.NET Numerics and `System.Numerics.Tensors`,
and nothing else - not a general "libraries are fine now" opening. The
rationale is narrow and specific: for the ops these two cover (matmul
above all), they're effectively faster wrappers around math this project
has *already implemented and understood* from first principles (TASK-003),
not new capability or a different mechanism. They don't replace the
hand-written scalar implementation - that stays the real, default,
always-correct reference, on by default. The opt-in fast path exists so
someone who has already been through the earlier stages (and understood
the mechanics of what a matmul actually does) has a faster way to get
through *later* stages - training a bigger model, faster - without
re-litigating that understanding or standing up a whole SIMD/BLAS
implementation by hand just to make later experimentation practical on
real hardware. See stage 9 below for the open design questions this still
needs to answer.

**A second explicit, opt-in exception (stage 11 / TASK-031 onward):**
[ILGPU](https://github.com/m4rs-mt/ILGPU) - a pure C# library that JIT-compiles
ordinary C# methods into GPU kernels (CUDA on NVIDIA hardware, OpenCL on
AMD/Intel/NVIDIA, or a CPU accelerator with no GPU at all), added at the
user's request specifically to demonstrate GPU-based training as part of
this project's lesson plan. No C/C++ interop code to write - ILGPU is the
interop layer. The rationale mirrors TASK-015's: this project still writes
and owns the actual kernel logic (matmul above all) itself, just targeting
a GPU execution model instead of a CPU loop - a different *backend* for
math already implemented and understood from first principles (TASK-003),
not a black-box library that does the math *for* this project the way
TorchSharp or ML.NET would. It doesn't replace the hand-written scalar
path (still the default, always-correct reference) or TASK-015's CPU fast
path - a third, independently-selectable backend. See stage 11 below for
the open design questions.

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

   **CLI gap, flagged at the user's request while auditing the README
   lesson plan (TASK-023) against what's actually runnable:** every other
   stage that produces or consumes a model artifact has a CLI (tokeniser
   training, chat), but pretraining a model from a corpus is library-only
   - there's no command to run, only a C# snippet to paste. → TASK-025.
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

   **Follow-up, flagged at the user's request while auditing the README
   lesson plan (TASK-023) once instruction tuning (TASK-016) existed, now
   done (TASK-027, TASK-028):** even fine-tuned via TASK-016, the chat CLI
   didn't apply the SFT prompt template per turn, had no notion of a
   response "finishing" (it free-ran to `--max-new-tokens` regardless), and
   didn't reformat multi-turn history with role markers the way training
   data was shaped. TASK-027 added an opt-in `--instruction-tuned` flag
   that wraps each turn in `SftDataset`'s own template, stops generation at
   the next `### Instruction:` marker instead of running on, and - since
   only the trimmed response text is ever appended back into history - every
   prior turn ends up template-shaped automatically. TASK-028 separately
   added `--context-length` so a conversation's window isn't stuck at
   just the model's own fixed `MaxSequenceLength`. Both remain CLI/runtime
   improvements only - the user/learner is still expected to supply their
   own pretraining corpus and SFT dataset beyond `examples/sft-example.jsonl`,
   same as every other stage.
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
   - This exception names exactly Math.NET Numerics and
     `System.Numerics.Tensors` - not "libraries are fine now" generally.
     The justification is that for the ops they'd cover, they're faster
     wrappers around math already implemented and understood from first
     principles in TASK-003, not new/different mechanics being hidden.
     They don't replace that implementation, only offer a faster route
     through *later* stages once the earlier ones are understood. Any
     other library (ML/tokeniser/tensor or otherwise) is still out of
     scope and would need its own explicit conversation, not "well we
     already made an exception once."
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
10. **Instruction tuning (SFT)** — continue training a pretrained
    checkpoint (TASK-012) on (instruction, response) example pairs instead
    of continuous raw text, so the resulting model actually behaves like
    something TASK-014's chat CLI can usefully talk to, rather than just
    continuing text in the training corpus's style. Added after the
    original 7-stage plan, at the user's request.

    **This is genuinely new capability, not just re-wiring existing
    pieces** - two real gaps, not implementation details to wave through:
    - **Data shape.** TASK-010's `BatchSampler` draws sliding windows from
      one continuous token stream - there's no such thing as an "example
      boundary" in that model. Instruction tuning data is inherently
      example-based (each instruction/response pair is its own training
      example), which needs a different data pipeline, not a mode switch
      on the existing one.
    - **Loss masking.** Standard SFT practice computes the loss only over
      the *response* tokens, not the instruction/prompt tokens - the model
      shouldn't be penalised for not "predicting" the user's own words.
      `CrossEntropyLoss.Compute` (TASK-011) has no masking concept today;
      it averages over every position.

    See TASK-016 for how these get resolved; this is deliberately left
    open here rather than pre-deciding an implementation.

    **Same CLI gap as stage 6, flagged at the same time:** fine-tuning a
    pretrained checkpoint on an SFT dataset is also library-only - no
    command to run. → TASK-026.

    **Where the (instruction, response) pairs themselves come from is a
    data-sourcing question, not a code dependency** - separate from the
    "no libraries" rule (that rule is about what code this program links
    against, not how a human prepares a training data file offline). A
    real attempt at this will likely combine more than one of:
    - **Manual authoring.** Most reliable, most labour-intensive. Best for
      a small (tens to low hundreds), high-quality, deliberately-scoped
      set covering a specific domain/style rather than broad generality -
      realistic for what a from-scratch, personal-machine-trained model
      can actually learn from anyway.
    - **Reformatting data that already exists.** FAQ pages, documentation,
      Q&A content, support transcripts, etc. the user has rights to use,
      rewritten into (instruction, response) shape. Semi-manual, but
      scales further than writing every pair from nothing.
    - **Generating candidate pairs with a separate, stronger model**, used
      purely offline as a data-preparation tool (e.g. prompting an
      existing assistant to draft instruction/response examples for given
      topics, or to draft responses for a list of instructions the user
      supplies) - the "self-instruct" style approach several public
      instruction-tuning datasets were bootstrapped with. This model
      (Claude, or whatever the user has access to) is not a runtime
      dependency of this project - it's a one-time, offline step producing
      a plain data file, same as if a human had typed it. Draft output
      from this route should be reviewed/edited before use, not accepted
      uncritically.
    - **Existing public instruction-tuning datasets** (e.g. Alpaca,
      Dolly) as a starting point or reference for format - mind licensing
      terms before use, and note that reformatting to whatever template
      TASK-016 settles on will still be needed.
11. **Optional GPU-accelerated backend (ILGPU)** — a third, opt-in
    `Tensor` backend alongside the existing hand-written scalar path
    (default) and TASK-015's CPU-optimised path, this time targeting a
    GPU instead of the CPU, so GPU-based training can genuinely be
    demonstrated as part of this project's lesson plan rather than only
    described. Added after the original plan, at the user's explicit
    request specifically for this purpose - not a performance initiative
    for its own sake.

    **Why ILGPU, and why this is scoped narrowly, mirroring stage 9:**
    - Pure C#: ILGPU JIT-compiles ordinary C# methods into GPU kernels at
      runtime. No C/C++ interop code to write by hand, and no native
      library binding to maintain - the one thing that made a GPU backend
      worth considering for a project whose whole premise is "no ML
      libraries, understand every mechanism." This project still writes
      the actual kernel logic (matmul above all) itself; ILGPU changes
      *where* that logic executes, not who wrote it - unlike TorchSharp or
      ML.NET, which would hide the actual computation behind a library
      call the way the "no libraries" rule exists to prevent.
    - Backend coverage: CUDA (NVIDIA), OpenCL (AMD, Intel, and NVIDIA as a
      fallback), and a CPU accelerator (no GPU needed at all, useful for
      correctness testing and CI where real GPU hardware isn't available).
      This project's own dev machine has a discrete AMD GPU (Radeon
      RX 6700 XT / Navi 22), so a genuine end-to-end GPU demo here would
      exercise ILGPU's OpenCL path, not CUDA - worth being explicit about
      in the README rather than writing docs that assume NVIDIA/CUDA by
      default.
    - Same threading mechanism as TASK-015's `Tensor.Backend` switch - a
      process-wide selection (e.g. extending the existing backend enum or
      a sibling switch), not a parameter threaded through every op or
      call site.
    - Same correctness bar as TASK-015: the GPU path must produce results
      equivalent (within float tolerance) to the existing scalar path,
      proven by running the *same* test suite (including gradient checks)
      against it - not a separate, smaller GPU-only test suite. Tests
      should default to ILGPU's CPU accelerator so the suite runs without
      requiring real GPU hardware to be present; a real-hardware run is a
      manual/demo step, not something CI can assume.
    - Open questions a real implementation needs to resolve, not
      pre-decided here: which ops actually move to the GPU (matmul first,
      per TASK-015's own priority reasoning: O(n³) dominates); how
      `IFloatBuffer`'s heap-vs-disk-backed storage split interacts with a
      GPU backend, which needs its own device-memory buffers regardless
      (a third storage location, not just a third compute path); and
      whether/how host↔device transfer overhead makes the GPU path a net
      win only above some model/batch size, which the demo should measure
      and report honestly rather than assume.
    - What this explicitly does **not** open up: no other GPU/ML library,
      no multi-GPU or distributed training, no mixed precision - this is
      "run the same math this project already understands, elsewhere,"
      not a general performance-library door.

## Documentation (TASK-023)
README.md is due a rewrite into a lesson plan: one section per stage
(mirroring the numbered list above), each covering what problem that stage
solves and what's actually happening conceptually, which source files to
read, and what to run to see it working - covering the full range from a
directory of raw `.txt` files through to a usable chatbot, including
stages not built yet (be clear about what's runnable today versus
planned, not presented as if it already works). Added at the user's
request; not yet started.

## Known limitations / deferred (not bugs - documented trade-offs)
Flagged individually in TASK.md as each was made; collected here for a
single view of what a real training run might need to revisit. All but
the last now have a follow-up task in TASK.md (added at the user's
request); distributed (multi-machine) training remains out of scope for
now (not planned/tasked, though revisitable if asked, unlike the "no ML
libraries" rule itself) - single-machine GPU execution is no longer in
that category, see stage 11 above and TASK-031 onward:

- **Bulk-encoding a large corpus is slow.** `BpeTokeniser.Encode` uses the
  simple merge-scan approach tuned for short/moderate text (TASK-010),
  not the efficient training-time algorithm `Train` uses. Fine for a
  prompt; would need work for encoding a full large corpus into training
  data. → TASK-018.
- **No KV-cache during generation** (TASK-013): every step recomputes a
  full forward pass over the whole context. Simple and correct, not fast.
  → TASK-020, the largest of the four follow-ups - a real architecture
  addition, not a small patch.
- **Checkpoints and AdamW's moment estimates are plain heap allocations**
  (TASK-011/012), not disk-backed like the tokeniser's/tensor engine's
  large structures - nothing built so far has been large enough to need
  it. → TASK-019, deliberately lower-priority: the mechanism is cheap to
  build (mirrors the existing `MappedArray<T>`/`ZerosOnDisk` pattern) but
  build it when a model is actually large enough to need it, not
  speculatively.
- **Two places don't use the max-subtraction softmax stability trick**:
  `Variable.Softmax` (TASK-004, used in attention/training - would need a
  max-along-axis Tensor reduction that doesn't exist yet) and
  `CrossEntropyLoss`'s `log(sum(exp(x)))` term (TASK-011). Fine at the
  scales tested so far; `Generation.TokenSampler`'s softmax *does* use the
  trick, since it has no backward pass to complicate and temperature
  scaling can push values to real extremes. → TASK-017, the smallest of
  the four follow-ups.
- **Regex-based pre-tokenisation** — the current whole-text byte stream
  approach works, but has no notion of "chunk" boundaries (word/whitespace/
  punctuation runs) that BPE merges shouldn't cross. → TASK-022. Explicitly
  *not* GPT-2's original splitting pattern - it's dated (e.g. its number
  handling and Unicode/whitespace treatment are worse than what came
  later); use a more modern pattern (GPT-4/`cl100k_base`-style splitting
  is the reference point) as the design target instead of reproducing an
  old pattern just because GPT-2 is the model this project's byte-level
  BPE approach was originally modelled on.
- **No CPU parallelism anywhere** - confirmed by inspection (no
  `Parallel`/`Task.Run`/threading in `src/`), not merely undocumented: every
  `Tensor` op, matmul above all, runs as a single-threaded scalar loop.
  This is a distinct gap from GPU execution (a different backend, stage 11
  below) or distributed training (still excluded, see below) - it doesn't
  need a GPU or another machine, just more of the cores already on this
  one, via .NET's own Task Parallel Library (`Parallel.For`/
  `Parallel.ForEach` - BCL, not a "no libraries" exception the way
  TASK-015 is, any more than `MemoryMappedFile` or `async`/`await` are).
  → TASK-021.
- **No GPU-accelerated backend** - every `Tensor` op runs on the CPU today
  (the hand-written scalar path, or TASK-015's Math.NET/
  `System.Numerics.Tensors` fast path), never a GPU, so GPU-based training
  can only be described, not demonstrated. → stage 11 above, TASK-031
  onward (ILGPU), added at the user's explicit request specifically to
  close this gap - not previously planned, and mixed precision/multi-GPU
  training remain out of scope even once single-GPU execution lands.
- Distributed (multi-machine) training — out of scope for now, not
  planned or tasked. Unlike GPU execution before this update, there's no
  standing user request driving this one, so it stays undone rather than
  becoming a stage/task - but it's a "not currently planned" position, not
  a permanent architectural ban the way "no ML libraries generally" is;
  revisit if the user asks, the same way GPU execution just did. One
  machine throughout today, whether that machine's `Tensor` ops run on its
  CPU (default, or TASK-021's parallel CPU path) or its GPU (stage 11).

Flagged while answering a question about scaling to a real (~230 MB, 250
book) corpus and dataset - both are genuine gaps at that scale, not
theoretical:

- **`LinkedTokenStream.Build` loaded the whole input file onto the heap**
  via `File.ReadAllText` before anything reached disk-backed storage - the
  "honest surprise" README.md's memory/disk footprint section documented
  (peak RAM scaled with input size in both `BpeTokeniser.Train` and
  `EncodeBulk`, not just disk). → TASK-029, now fixed: `Build` streams each
  file via `PreTokeniser.Split(TextReader, bufferSize)` in two passes
  (count exact bytes, then fill) instead of holding it all in memory at
  once. Re-measured at 2/4/10/100 MB in README.md's footprint section -
  meaningfully lower peak RAM at every size (roughly halved at 10 MB), and
  no longer scaling anywhere near as steeply; what residual growth remains
  is the disk-backed arrays' own resident *write* working set (matches the
  existing 16-bytes-per-input-byte scratch formula), which is reclaimable
  by the OS unlike the old heap string, not a leftover version of the same
  bug.
- **The SFT CLI's `--steps`/`--batch-size` didn't scale to a real dataset.**
  The 6-example demo worked by setting `--batch-size` equal to the dataset
  size; a dataset of hundreds or thousands of (instruction, response)
  pairs needed actual epoch-based training (shuffled passes over the whole
  dataset) instead of a fixed step count the user has to hand-compute
  against dataset size themselves. → TASK-030, now fixed: `--epochs`
  (default 3) is the primary way to size a run, each epoch a freshly
  shuffled full pass (`SftTrainer.RunEpochs`); `--steps` stays as a
  lower-level, unshuffled escape hatch, mutually exclusive with `--epochs`.
  The demo now uses `--epochs 300` with the new fixed default batch size
  (8) instead of hand-setting `--batch-size` to match the dataset size.
