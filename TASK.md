# TASK.md

Standing constraint on every task below (see PLAN.md "Memory constraint"):
use disk as scratch space and minimise RAM usage for anything that scales
with corpus/model/batch size. Prefer disk-backed/streamed structures over
large heap allocations — this machine has OOM-killed itself on this project
before. Flagged explicitly on the tasks it matters most for below.

- [x] TASK-001: Byte-level BPE tokeniser (train, encode, decode, save/load)
  Required by: TASK-022
- [x] TASK-002: Directory input + disk-backed scratch for large corpora
  Required by: TASK-018

## Tensor + autodiff engine
Required by: TASK-005, TASK-006, TASK-007, TASK-008, TASK-009

- [x] TASK-003: `Tensor` type — N-dimensional float array (shape, strides,
  indexing) with elementwise ops (add, subtract, multiply, divide) and
  matmul, transpose, sum/mean-along-axis, broadcasting. Backing storage
  should be swappable between an in-heap array (small tensors) and a
  disk-backed store (large ones) so later tasks aren't locked into
  unbounded heap growth — see `src/Tokeniser/MappedInt32Array.cs` for the
  established pattern. Done: `src/Tensor/`, backed by the shared
  `Common.MappedArray<T>` (extracted from the tokeniser's
  `MappedInt32Array` as part of this task, now used by both). Ops always
  produce heap-backed results for now — routing large op results to disk
  is left to whichever later task first needs it.
  Depends on: none
  Required by: TASK-015, TASK-021
- [x] TASK-004: Reverse-mode autodiff — computation graph recording ops as
  they run, `Backward()` to propagate gradients, gradient accumulation.
  Covers the ops from TASK-003 plus softmax, exp/log, and the activation
  functions transformer blocks need (GELU or ReLU). Done: `src/Tensor/Variable*.cs`
  — chose ReLU over GELU (PLAN.md's "either" framing) to keep the backward
  math simple; softmax has no max-subtraction stability trick yet (no
  max-along-axis reduction in Tensor), noted as a revisit candidate if large
  logits cause overflow. Tested via finite-difference gradient checking
  (`tests/Tensor.Tests/VariableTests.cs`) rather than hand-derived expected
  values, plus explicit tests for gradient accumulation (a variable used
  twice, and a diamond dependency).
  Depends on: TASK-003
  Required by: TASK-017

## Embeddings
Required by: TASK-008

- [x] TASK-005: Token embedding table — learned lookup from token id to a
  dense vector, backed by `Tensor`, trainable via the autodiff engine.
  Done: `src/Model/TokenEmbedding.cs`, on a new differentiable
  `Tensor.GatherRows`/`ScatterAddRows` pair (`Variable.GatherRows`) added
  to the tensor engine for this — row lookup with gradient accumulation on
  repeated indices, which is what makes a token appearing twice in one
  sequence get the sum of both occurrences' gradients rather than losing
  one. Weights init as small Gaussian noise (mean 0, std 0.02, GPT-2's
  scale) via a hand-rolled Box-Muller transform, not zeros — identical
  rows would otherwise get identical gradients forever.
  Depends on: TASK-004
- [x] TASK-006: Positional encoding — learned positional embeddings added to
  token embeddings (start with learned rather than sinusoidal/RoPE: fewer
  moving parts for a first pass). Done: `src/Model/PositionalEmbedding.cs`
  — same shape as `TokenEmbedding` (a `[maxSequenceLength, embeddingDim]`
  trainable table, looked up via the same `GatherRows` op), sized to a max
  sequence length rather than generalising to arbitrary lengths like
  sinusoidal/RoPE would. Shared the Gaussian weight-init logic with
  `TokenEmbedding` via a new `GaussianInit` helper instead of duplicating
  it. Composition with `TokenEmbedding` (plain elementwise `Add`, both
  produce `[sequenceLength, embeddingDim]`) is covered by tests but not
  wrapped in its own class yet - left for whichever later task first
  assembles a full embedding layer.
  Depends on: TASK-004

## Attention + transformer block
Required by: TASK-008

- [x] TASK-007: Scaled dot-product attention + multi-head attention.
  Done: `src/Model/ScaledDotProductAttention.cs` (parameter-free,
  batched over any leading dims so it takes numHeads as a batch
  dimension) and `src/Model/MultiHeadAttention.cs` (owns the Q/K/V/O
  projection weights, splits into heads via `Reshape`+`Transpose`,
  concatenates back). Includes causal masking (additive -infinity mask
  before softmax) since the model this feeds is decoder-only/autoregressive
  — without it, training would let a position "see" the token it's meant
  to predict. No projection bias terms (kept to what's needed; feed-forward
  in TASK-008 needs bias, attention doesn't strictly). Added
  `Variable.Reshape` (mirrors `Tensor.Reshape`) to the autodiff engine to
  support head-splitting. Tested via finite-difference gradient checks
  (including against persistent weights already embedded in a module, a
  different shape of check than TASK-004's) and explicit causal-masking
  correctness tests (changing a future position's value must not change an
  earlier position's output; contrasted against non-causal where it does).
  Depends on: TASK-004
  Required by: TASK-020
- [x] TASK-008: Full transformer block — multi-head attention, feed-forward
  layer, layernorm, residual connections, assembled and unit-tested as one
  block with a known input/output shape. Done:
  `src/Model/{LayerNorm,FeedForward,TransformerBlock}.cs`, GPT-2-style
  pre-norm layout (normalise -> attend -> +residual -> normalise ->
  feed-forward -> +residual). FeedForward expands to `4*embeddingDim`
  hidden units by default (GPT-2's convention), ReLU in between (per
  TASK-004's choice over GELU). Added `Tensor.Sqrt`/`Variable.Sqrt` to the
  tensor engine for layernorm's variance normalisation. Tested end to end:
  causal masking survives being wrapped in a full block, gradient checks
  against a representative parameter in every sub-layer (not just one),
  and a per-position-independence check for both layernorm and
  feed-forward (no cross-position mixing outside of attention).
  Depends on: TASK-005, TASK-006, TASK-007
  Required by: TASK-020

## Model assembly
- [x] TASK-009: Decoder-only (GPT-style) model — stack N transformer blocks
  from TASK-008, add the output projection back to vocabulary logits over
  the tokeniser's vocabulary (TASK-001). Done: `src/Model/GptModel.cs` —
  token+positional embeddings, N causal `TransformerBlock`s, a final
  layernorm (GPT-2's "ln_f"), then an output projection that reuses
  (rather than duplicates) `TokenEmbedding.Weight`, transposed - weight
  tying, same as GPT-2 itself. Tested end to end: causal masking survives
  the full model (a later token can't change an earlier position's
  logits), weight tying verified by checking the output projection's
  gradient reaches embedding rows that were never looked up as input
  tokens (proof it's the same matrix, not just equal by coincidence), and
  finite-difference gradient checks against parameters spanning every
  stage (token/positional embeddings, a block's attention weight, the
  final norm).
  Depends on: TASK-008
  Required by: TASK-020

## Training loop
- [x] TASK-010: Batching — turn a tokenised corpus (TASK-001/002 output)
  into fixed-length training batches (input/target pairs shifted by one
  token) for next-token prediction. Stream/index batches from the
  disk-backed token store rather than loading the full tokenised corpus
  into RAM at once. Done: new `src/Training/` project —
  `TokenCorpus` (a token-id stream in a disk-backed `MappedArray<int>`,
  reusing `Common`'s pattern) and `BatchSampler` (draws fixed-length
  input/target windows from it; batch memory is O(batchSize*contextLength),
  never O(corpus length)). `GptModel` (TASK-009) has no batch dimension of
  its own, so a "batch" here is just a set of examples for TASK-012's
  training loop to accumulate gradients over, not a single batched tensor
  op. Known gap: getting token ids into `TokenCorpus` in the first place
  still goes through `BpeTokeniser.Encode`, which (unlike `Train`) uses the
  simple merge-scan approach tuned for short/moderate text, not efficient
  bulk encoding of a large corpus - noted here rather than solved, since
  it's outside batching itself; revisit if an actual training run proves
  it's too slow in practice.
  Depends on: TASK-002
- [x] TASK-011: Cross-entropy loss + optimizer (SGD first, then AdamW) built
  on the autodiff engine. Done: `src/Training/{CrossEntropyLoss,SgdOptimizer,AdamWOptimizer}.cs`.
  Loss computes log-softmax directly (x - log(sum(exp(x)))) rather than
  Softmax().Log(), to avoid round-tripping through a softmax value that
  could already have underflowed to 0 - same no-max-subtraction caveat as
  `Variable.Softmax` still applies to the sum(exp(x)) term itself, though.
  Needed two new tensor primitives: `Tensor.GatherColumns`/`ScatterAddColumns`
  (pick one column per row - "the log-probability assigned to the correct
  next token" - and its gradient-scattering inverse), and
  `Tensor.SubtractInPlace`, the one deliberate mutation primitive in the
  whole tensor engine (every other op returns a new tensor): needed
  because a `Variable.Value` can't be reassigned once an optimizer holds
  that Variable by reference. AdamW's moment-estimate state is plain heap
  tensors for now (~2x parameter count); TASK-012 decides whether that
  needs disk-backing once model size makes it non-trivial (already flagged
  there in PLAN.md). Tested via known-exact-value checks (uniform logits
  give loss = log(vocabSize) exactly), gradient-direction sanity, finite-
  difference gradient checks, a hand-computed single AdamW step, and two
  end-to-end convergence tests (minimising a toy quadratic, and driving
  down cross-entropy loss against fixed targets over many steps).
  Depends on: TASK-004
  Required by: TASK-016, TASK-017, TASK-019
- [x] TASK-012: Training loop — wires TASK-009/010/011 together: forward
  pass, loss, backward pass, optimizer step, checkpointing (save/load model
  weights), basic logging of loss over time. Checkpoints and optimizer
  state (e.g. Adam's moment estimates, which double memory vs. the raw
  weights) are candidates for disk-backed storage once model size makes
  them non-trivial. Done: `src/Training/{Trainer,ModelCheckpoint}.cs`, plus
  a new `GptModel.Parameters()` (and matching `Parameters()` on every
  sub-layer, composed bottom-up) so an optimizer/checkpoint can enumerate
  every trainable Variable - explicitly excludes the weight-tied output
  projection to avoid double-counting it. `Trainer.Step` accumulates
  gradients over a batch by scaling each example's loss by 1/batchSize
  before its own `Backward()` call (rather than summing raw gradients and
  scaling after), keeping the effective step size independent of batch
  size using only existing autodiff ops. Logging is a plain
  `Action<int,float>? onStep` callback (mirrors `BpeTokeniser.Train`'s
  `onMerge`), not hardcoded console output. Checkpointing is binary
  (shape + values per parameter, hyperparameters in the header so `Load`
  reconstructs the exact architecture before overwriting values in place)
  - deferred rather than solved: still plain heap allocation, no
  disk-backing yet for checkpoints or AdamW's moment estimates, since
  nothing in this project has been large enough yet to need it; revisit
  if that changes. Tested via checkpoint round-trip (identical parameter
  values and identical forward-pass output before/after save+load,
  corrupted-file rejection) and a genuine end-to-end training run: loss
  measurably drops on a small repetitive corpus over 100 steps, not just
  a unit-level check of each piece in isolation.
  Depends on: TASK-009, TASK-010, TASK-011
  Required by: TASK-014, TASK-016

## Generation
- [x] TASK-013: Sampling — greedy, temperature, and top-k/top-p decoding
  from a trained (or in-progress) model, using the tokeniser to decode
  output token ids back to text. Done: new `src/Generation/` project
  (`SamplingOptions`, `TokenSampler`, `TextGenerator`). Deliberately plain
  float-array math, not Tensor/Variable - sampling is inference-only, no
  gradient needed, so building an autodiff graph for it would be pure
  waste. Unlike `Variable.Softmax`, this softmax *does* use the
  max-subtraction stability trick (flagged as a gap back in TASK-004):
  no backward pass to complicate here, and temperature scaling can push
  logits to genuine extremes, so it costs nothing and actually matters.
  No KV-cache - every generation step recomputes a full forward pass over
  the whole context (simple and correct over fast; a production inference
  server would want one, this doesn't need one). Once the growing sequence
  would exceed the model's context window, generation keeps going via a
  sliding window over the most recent `MaxSequenceLength` tokens rather
  than stopping. Tested via deterministic greedy generation, statistical
  checks that sampled frequencies roughly track softmax probabilities and
  that top-k=1/very-small-top-p both collapse to near-greedy behaviour,
  and an end-to-end run through a real trained `BpeTokeniser` confirming
  the decoded output always starts with the decoded prompt exactly (true
  by construction, since decode is just concatenation in token order).
  Depends on: TASK-012
  Required by: TASK-014, TASK-020, TASK-023

## Interactive CLI
Added after the original 7-stage plan, at the user's request (not part of
TASK-001..013's original scope).

- [x] TASK-014: Interactive chat CLI — a console app that loads a saved
  `ModelCheckpoint` and tokeniser vocab (`BpeTokeniser.Load`), then loops:
  read a line of user input, generate a continuation via
  `Generation.TextGenerator`, print it, repeat. Maintains conversation
  state as a growing token-id sequence (not by re-encoding the whole
  transcript as text each turn) so multi-turn context actually accumulates
  correctly through `TextGenerator`'s existing sliding-window handling once
  it exceeds `MaxSequenceLength`. Sampling parameters (temperature, top-k,
  top-p, max new tokens per turn) configurable via CLI flags, matching the
  existing Tokeniser CLI's `--flag value` style. An exit command (e.g.
  `/exit` or Ctrl+C) to leave the loop.

  Needs to be honest in its own UI text (not just PLAN.md) that this is a
  raw next-token-prediction model, not an instruction-tuned assistant - it
  will continue text in the style of whatever it was trained on, not
  necessarily answer questions or follow instructions, unless the training
  corpus was itself chat-shaped. Don't oversell "chatbot" in the CLI's own
  banner/help text.

  Done: new `src/Chat/` project. Followed the same testable-CLI pattern
  established for TASK-024 from the start (`ChatCli.Run(args, stdin,
  stdout, stderr)`, `Program.cs` a one-line call) rather than writing it
  as untestable top-level statements and retrofitting later - `Chat.ChatCli`
  is at 100% branch coverage. The honest disclaimer appears in both the
  no-args usage text and the post-load banner, not just one or the other.
  Verified with a real (if tiny and untrained) checkpoint + vocab fixture,
  not just mocked: multi-turn conversations, all four sampling flags
  together, a corrupted-checkpoint error path (reusing the byte-corruption
  technique from `ModelCheckpointTests`), and deterministic output at
  temperature 0 across repeated runs.
  Depends on: TASK-012, TASK-013

## Optional optimised math backend
Added after the original 7-stage plan, at the user's request (not part of
TASK-001..013's original scope). This is the one deliberate, narrowly-scoped
exception to PLAN.md's "no libraries" rule, scoped to exactly two named
libraries - Math.NET Numerics and `System.Numerics.Tensors` - not a general
opening. The justification: for the ops they'd cover (matmul above all),
they're faster wrappers around math this project already implemented and
understood from first principles in TASK-003, not new/different mechanics
being hidden behind a library call. They don't replace the hand-written
implementation, only offer a faster route through *later* stages (bigger
training runs) once the earlier ones are understood. See PLAN.md stage 9
for the full rationale and open design questions before starting this task.

- [ ] TASK-015: `--optimised` opt-in fast path for `Tensor`'s hot ops
  (matmul first and foremost; elementwise ops as a stretch goal), backed by
  Math.NET Numerics and/or `System.Numerics.Tensors` only - no other
  library. Off by default - the existing hand-written scalar implementation
  remains the default, always-correct reference implementation, not
  something this task replaces or deletes. Must keep the standing
  memory-discipline constraint intact (PLAN.md): the optimised path most likely only applies to
  heap-backed tensors (`HeapFloatBuffer`), since the library types involved
  expect contiguous managed memory, not `MappedArray<T>`'s disk-backed
  storage - large disk-backed tensors should keep using the scalar path,
  or an explicitly-scoped chunked approach if one is worked out, not
  silently lose their memory-safety guarantees. Needs a way to select the
  backend process-wide (e.g. from a CLI flag down into `Tensor`) without
  adding a parameter to every op's call site. Correctness must be
  demonstrated by running the *existing* test suite - including the
  finite-difference gradient checks in `Tensor.Tests`/`Model.Tests` - against
  both backends and getting equivalent results, not a separate smaller
  test suite scoped to just the fast path.
  Depends on: TASK-003

## Instruction tuning
Added after the original 7-stage plan, at the user's request (not part of
TASK-001..013's original scope). Pairs naturally with TASK-014 (a fine-tuned
checkpoint is what makes the chat CLI's output actually useful) but doesn't
depend on it - this can be built and tested independently of the CLI.

- [ ] TASK-016: Instruction tuning (SFT) - continue training a pretrained
  `GptModel` checkpoint on (instruction, response) example pairs, with the
  loss restricted to response tokens only. Two genuinely new capabilities
  needed, not just re-wiring TASK-010/011/012 (see PLAN.md stage 10 for the
  full rationale):
  - **An example-based dataset**, distinct from TASK-010's `TokenCorpus`/
    `BatchSampler` (continuous-stream sliding windows). Needs a data format
    decision (a plain-text template with a fixed instruction/response
    delimiter is the simplest starting point; a structured format like
    JSON Lines is the more extensible one - pick one, don't build both) and
    a loader that tokenises each pair into (inputIds, targetIds) plus which
    target positions are actually response tokens.
  - **Masked cross-entropy loss** - extend or parallel
    `Training.CrossEntropyLoss` to average only over positions flagged as
    "response," not every position, unlike TASK-011's version.
  Reuses: `ModelCheckpoint.Load` to start from a pretrained model (not
  random init) and `ModelCheckpoint.Save` for the fine-tuned result as a
  *separate* checkpoint file (never overwrite the base pretrained one);
  the same optimizer/backward-pass machinery as `Trainer`, likely via a
  generalised `Trainer` or a sibling class - which one is an open design
  question, not decided here. Conventionally uses a smaller learning rate
  than pretraining; worth a sensible default plus making it configurable
  rather than hardcoding pretraining's rate.

  Sourcing the (instruction, response) pairs is a data-preparation
  question, separate from the "no libraries" rule (that's about runtime
  code dependencies, not how a data file gets written offline) - see
  PLAN.md stage 10 for the fuller version. In short: manual authoring for
  a small high-quality set, reformatting existing owned content (docs,
  FAQs), generating draft pairs with a separate stronger model used purely
  as an offline one-time data-prep tool (review before use, don't accept
  uncritically), and/or existing public instruction-tuning datasets
  (mind licensing) reformatted to whatever template this task settles on.
  Depends on: TASK-011, TASK-012

## Known limitations follow-ups
Added at the user's request: one follow-up task per item in PLAN.md's
"Known limitations / deferred" list (except GPU/distributed training,
flagged there as genuinely out of scope, not just undone). Ordered
smallest to largest, TASK-021/022 excepted (added later, appended at the
end rather than re-numbered into size order).

- [ ] TASK-017: Softmax numerical stability - add `Tensor.Max(axis,
  keepDims)` (mirrors the existing `Sum`/`Mean` reductions in
  `Tensor.Reductions.cs`), then subtract the per-row max before `Exp()` in
  both `Variable.Softmax` (TASK-004) and `CrossEntropyLoss`'s
  `log(sum(exp(x)))` term (TASK-011) - the standard "safe softmax" trick.
  The smallest of these four: one new Tensor op plus two call-site
  updates, no architecture change. `Generation.TokenSampler`'s softmax
  already does this (it has no backward pass to complicate), so it's a
  useful reference for the expected behaviour, not something to touch.
  Verify via the existing finite-difference gradient checks (result should
  be numerically identical for inputs that don't overflow either way) plus
  a new test using large-magnitude logits that would overflow/underflow
  without the fix.
  Depends on: TASK-004, TASK-011

- [ ] TASK-018: Fast bulk-encode for `BpeTokeniser` - a second encode path
  that *applies* an already-learned merge table (no new merges learned,
  unlike `Train`) but reuses `Train`'s disk-backed intrusive-linked-list
  approach instead of `Encode`'s simple repeated merge-scan, so it scales
  to a full large corpus instead of just a short prompt. This is what
  would let `TokenCorpus` (TASK-010) be populated directly from a large
  corpus at training time, rather than the current slow per-call `Encode`
  path. Needs its own correctness tests (same output as `Encode` for
  inputs `Encode` can still handle, just faster) rather than assuming
  parity with `Train`'s different algorithm.
  Depends on: TASK-002

- [ ] TASK-019: Optional disk-backed AdamW moment estimates - a
  constructor flag (e.g. `useDiskBackedState` + a scratch directory) on
  `AdamWOptimizer` backing `m`/`v` with `Tensor.ZerosOnDisk` instead of
  `Zeros`, mirroring the existing `MappedArray<T>` pattern. Deliberately
  lower priority than the other three: cheap to build when the time comes,
  but nothing built in this project so far has been large enough to need
  it - build this when a real model size makes it matter, not
  speculatively ahead of that.
  Depends on: TASK-011

- [ ] TASK-020: KV-cache for generation - the largest of these four, a
  real architecture addition rather than a small patch. Needs a stateful
  "generation session" concept holding cached K/V per layer, a modified
  single-token forward path threaded through `MultiHeadAttention`,
  `TransformerBlock`, and `GptModel` (today's `Forward` always recomputes
  every layer over the whole context), and `Generation.TextGenerator`
  updated to use the cached path during autoregressive generation instead
  of re-forwarding the whole growing sequence every step. Correctness is
  the critical thing to prove here, not just speed: cached-path output
  must be *exactly* the same as today's non-cached output for the same
  token sequence, at every step, not just "plausibly similar" - test by
  comparing the two paths directly, not only by inspecting generated text.
  Depends on: TASK-007, TASK-008, TASK-009, TASK-013

- [ ] TASK-021: CPU parallelism for `Tensor`'s hot ops - unlike TASK-015,
  this doesn't touch the "no libraries" question at all: .NET's Task
  Parallel Library (`Parallel.For`/`Parallel.ForEach`, `System.Threading.Tasks`)
  is BCL, not a new dependency, and the algorithm doesn't change - only how
  many of this machine's cores execute the *same* hand-written scalar loop.
  Confirmed via inspection that nothing in `src/` currently uses any
  parallelism, so this is a real, not hypothetical, gap on a 16-core/
  32-thread machine. Priority mirrors TASK-015: matmul's outer loops
  (batch, and/or the output-row loop within each matmul) first - it's the
  O(n³) dominant cost in a forward/backward pass - elementwise ops (`Tensor.
  Elementwise.cs`/`Tensor.Unary.cs`) as a lower-priority follow-on given
  their O(n) cost is far smaller. **Must preserve determinism**: only
  parallelise independent outer loops (separate output rows/batches/
  elements), never the inner accumulation loop within a single matmul
  output element (`Tensor.MatMul`'s `for p in k` reduction) - a parallel
  reduction there would make floating-point summation order (and therefore
  the exact result, though not its correctness) non-deterministic across
  runs, breaking bit-for-bit comparison against the existing test suite's
  expected values. Likely not worth it for small tensors (thread-scheduling
  overhead exceeds the work, e.g. the many `[1]`-shaped scalar tensors used
  throughout this codebase for constants) - a size threshold below which
  the existing sequential path is kept is worth considering, not just
  parallelising unconditionally. Verify via the *existing* test suite
  (results must match, not just "look about right") plus a manual/
  benchmark wall-clock comparison - performance shouldn't be asserted in
  the automated suite, environments vary too much for that to be reliable.
  Depends on: TASK-003

- [ ] TASK-022: Modern regex-based pre-tokenisation for the BPE tokeniser -
  split text into chunks (word/whitespace/punctuation-ish runs) *before*
  BPE merge-learning and encoding, so merges never cross a chunk boundary
  in ways that produce bad tokens (e.g. merging a trailing space into a
  word, or digits into overly-long number tokens). Explicitly **not**
  GPT-2's original splitting regex - it's dated (weaker Unicode handling,
  and no cap on how many digits a number chunk can span, among other
  issues later tokenisers fixed) - use a more modern pattern as the
  reference (GPT-4/`cl100k_base`-style splitting is the design target) via
  `System.Text.RegularExpressions` (BCL, same non-issue as TASK-021's use
  of TPL). Needs to apply identically in both `BpeTokeniser.Train` (pre-split
  each input the same way file boundaries already prevent cross-document
  merges - chunk boundaries become another place merges must never cross)
  and `BpeTokeniser.Encode` (split first, encode each chunk independently,
  concatenate) - training and inference must use the same chunking or
  results silently diverge. **Breaking change to the token format**:
  existing saved `vocab.bpe` files and anything trained against them
  (checkpoints, since `vocabSize`/token ids shift) become incompatible and
  need retraining - call this out clearly wherever it lands (README,
  commit message), not just here. Test that merges never cross a chunk
  boundary, and that encode/decode stays an exact roundtrip through the
  new chunking.
  Depends on: TASK-001

## Documentation
Added at the user's request.

- [ ] TASK-023: Rewrite README.md as a lesson plan - walk through the whole
  process from a directory of raw `.txt` files all the way to a usable
  chatbot, one section per stage (mirroring PLAN.md's stage list), each
  covering: what problem that stage solves and what's actually happening
  conceptually, which source files to go read, and what to run to see it
  working (a CLI command where one exists, otherwise a code snippet or the
  relevant test project). Covers the full range including stages not built
  yet (TASK-014 chat CLI, TASK-016 instruction tuning) - be clear about
  what's runnable today versus what's planned, don't present unbuilt stages
  as if they already work.
  Depends on: TASK-013

## Test coverage
Added at the user's request alongside AGENTS.md, which sets the 90%
branch-coverage bar this task exists to reach.

- [x] TASK-024: Raise branch coverage to >=90% across every test project.
  **Methodology correction made while doing this task**: measuring each
  test project's own cobertura report in isolation (its top-level
  `branch-rate`) is the wrong metric - it includes every assembly that
  project *references*, not just its own, so e.g. `Generation.Tests`
  showed ~57% mostly because it doesn't itself exercise most of `Model`'s
  or `Tensor`'s surface (those are `Model.Tests`'/`Tensor.Tests`' job).
  The metric that actually matters is: does each *production* project's
  own code reach 90%+ branch coverage from the whole test suite combined
  (all test projects that touch it, merged) - measured with
  `reportgenerator` (`dotnet tool install -g dotnet-reportgenerator-globaltool`)
  merging every test project's coverage output. By that measure every
  production assembly is now at 92–100% branch coverage (solution-wide:
  97.2%, 420/432 branches).
  Done via, per assembly: new `tests/Common.Tests/` (didn't exist at all -
  `MappedArray<T>`'s double-`Dispose()` guard was untested); a `Random?
  random = null` default-argument branch untested across most of `Model`'s
  constructors (added a "constructed without an explicit Random" test to
  each); a real, previously-undiscovered untested error path in
  `ModelCheckpoint.Load` (a corrupted parameter shape now throws, not just
  a corrupted count); several `Tensor` validation branches (negative
  indices, wrong rank, axis/dimension bounds) that only had the
  "too-large" half of their range-check tested, not the negative half;
  `TokenSampler`'s top-k/top-p edge cases (k<=0, k>=length, a threshold
  requiring multiple candidates). Biggest single gap:
  `src/Tokeniser/Program.cs` was 0% covered, being top-level statements
  (uninvokable from a test project) - extracted its logic into a new,
  directly-testable `TokeniserCli.Run(args, stdout, stderr)` (output
  streams as parameters so tests can capture them), with `Program.cs`
  reduced to a one-line call. That surfaced (and let us then unit test)
  the disk-budget check's arithmetic, which had never been exercised at
  all - extracted into a small pure `ExceedsDiskBudget`/`EstimateScratchBytes`
  pair for direct testing, same pattern as the pre-existing `IsTmpfs`.
  A few branches remain genuinely impractical to hit without further
  dependency-injection (a non-Linux `OperatingSystem.IsLinux()` check, and
  malformed-line handling in live `/proc/mounts` parsing) - left as-is
  rather than over-engineering a seam for them.
  Depends on: none

## Notes
- Tasks are scoped for hand-rolled, no-library implementation per PLAN.md,
  except TASK-015, which is an explicit, narrowly-scoped, opt-in exception.
- Work through tasks one at a time in order; ambiguities get clarified
  before implementation starts on that task, not up front for all of them.
