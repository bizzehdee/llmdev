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

- [x] TASK-015: `--optimised` opt-in fast path for `Tensor`'s hot ops
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
  Done: chose `System.Numerics.Tensors` (`TensorPrimitives.Dot`) over
  Math.NET - zero extra package-manager surface beyond the one NuGet
  reference, and it's a direct SIMD analogue of the scalar dot-product
  loop matmul already does, matching the "faster wrapper around code we
  already have" framing. `IFloatBuffer.TryGetSpan` added so a buffer can
  opt into (heap) or decline (disk-backed) the fast path per-tensor;
  `TensorBackend` (`Scalar`/`Optimised`) selected process-wide via
  `Tensor.Backend`, an `AsyncLocal<T>`-backed static (not a plain static,
  to avoid state leaking between concurrently-running xUnit tests or,
  later, real parallel work from TASK-021) defaulting to `Scalar`.
  `Tensor.MatMul.cs` dispatches to the original scalar triple-loop
  (renamed `MatMulScalar`) or `MatMulOptimised` when both operands can
  hand out a contiguous span; `MatMulOptimised` transposes the right-hand
  operand once (always yields a heap-backed, span-capable result via
  `Zeros()`) so both operands' rows are contiguous, then calls
  `TensorPrimitives.Dot` per output element instead of a scalar
  accumulation loop. `src/Chat/ChatCli.cs` gained a `--optimised` flag
  that sets `Tensor.Backend` before the conversation loop starts.
  Verified per the task's own bar: the six existing `TensorTests.cs`
  MatMul tests and the three `VariableTests.cs` MatMul gradient-check
  tests were parametrised by `TensorBackend` (`[Theory]` +
  `[InlineData]`) rather than duplicated, plus a new test proving a
  disk-backed operand still falls back to the scalar path and stays
  correct even with `Backend == Optimised`. Solution-wide branch coverage
  held at 97.2% after the change (every production assembly still ≥90%).
  Depends on: TASK-003
  Required by: TASK-031

## Instruction tuning
Added after the original 7-stage plan, at the user's request (not part of
TASK-001..013's original scope). Pairs naturally with TASK-014 (a fine-tuned
checkpoint is what makes the chat CLI's output actually useful) but doesn't
depend on it - this can be built and tested independently of the CLI.

- [x] TASK-016: Instruction tuning (SFT) - continue training a pretrained
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
  Done: chose JSON Lines over a plain-text template - the more extensible
  of the two named options, and avoids needing escaping rules of its own
  the moment an instruction/response contains whatever delimiter a
  plain-text template would use. `Training.SftExample`/`SftTokenizedExample`
  (`src/Training/SftExample.cs`) and `Training.SftDataset`
  (`src/Training/SftDataset.cs`): `Load` reads one JSON object per
  non-blank line; `Tokenize` wraps the instruction in a fixed
  `"### Instruction:\n{0}\n\n### Response:\n"` prompt template and
  tokenises the templated prompt and the response *separately* (not the
  whole string in one `Encode` call), guaranteeing an exact token-level
  split regardless of where BPE merges would otherwise fall across that
  boundary; `ResponseMask[i]` is true iff position i's *target* (the
  standard next-token shift) falls at or past where the prompt's tokens
  end. `CrossEntropyLoss.ComputeMasked` (refactored the shared
  log-softmax-at-target-tokens computation out of `Compute` into a
  private `TargetLogProbs` helper both now call) zeroes out non-response
  positions before summing and divides by the *masked* count, not the
  total position count - averaging over everything would dilute the loss
  whenever a prompt is long relative to its response. `Training.SftTrainer`
  (`src/Training/SftTrainer.cs`) is a sibling to `Trainer`, not a
  generalisation of it - deliberate design call: the two operate over
  genuinely different data shapes (continuous-stream sliding windows vs.
  standalone instruction/response sequences) and different losses, so
  sharing an abstraction would cost more in indirection than it would
  save in duplication. Advances sequentially through the (typically
  small, curated) dataset rather than resampling randomly like
  `BatchSampler` does for a large pretraining corpus; gradient-accumulates
  over a batch the same way `Trainer.Step` does. Doesn't touch
  checkpointing itself (same as `Trainer`) - callers construct the model
  via `ModelCheckpoint.Load` and save the fine-tuned result to a separate
  path via `ModelCheckpoint.Save`; the learning-rate guidance (a tenth or
  less of pretraining's) is documented on the class, configured on the
  `IOptimizer` passed in. Tested: `SftDataset` tokenisation/masking
  correctness (mask is a contiguous true suffix, JSON Lines parsing,
  malformed-line handling) using a real trained `BpeTokeniser`;
  `CrossEntropyLoss.ComputeMasked` (scalar shape, masked-value-only
  effect on the loss, gradient only reaching masked positions, the
  no-masked-positions guard); an end-to-end `SftTrainer` test proving
  loss actually drops substantially on a small repetitive pattern,
  mirroring `TrainerTests`' pretraining equivalent. Solution-wide branch
  coverage held (`Training` 98.8%).
  Depends on: TASK-011, TASK-012
  Required by: TASK-030

## Known limitations follow-ups
Added at the user's request: one follow-up task per item in PLAN.md's
"Known limitations / deferred" list (except GPU/distributed training,
flagged there as genuinely out of scope, not just undone). Ordered
smallest to largest, TASK-021/022 excepted (added later, appended at the
end rather than re-numbered into size order).

- [x] TASK-017: Softmax numerical stability - add `Tensor.Max(axis,
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
  Done: `Tensor.Max` added in `src/Tensor/Tensor.Reductions.cs`.
  `Variable.Softmax` now subtracts `Value.Max(axis, keepDims: true)`
  before `Exp` - purely a forward-pass change, since the backward pass
  is expressed in terms of the already-computed softmax output, not the
  subtraction. `CrossEntropyLoss.Compute` now computes logsumexp via the
  shifted form (`max + log(sum(exp(x - max)))`), with the max wrapped as
  a constant `Variable` (no parent op) to deliberately stop gradient
  flowing through the max itself - correct since d(logsumexp)/dx_i =
  softmax(x)_i regardless of the shift. Added large-magnitude-logit tests
  (finite output, correct sum-to-one / shift-invariance) to
  `VariableTests.cs` and `CrossEntropyLossTests.cs`, plus `Tensor.Max`
  tests in `TensorTests.cs`. All existing finite-difference gradient
  checks still pass unchanged; solution-wide branch coverage held at
  97.4%.
  Depends on: TASK-004, TASK-011

- [x] TASK-018: Fast bulk-encode for `BpeTokeniser` - a second encode path
  that *applies* an already-learned merge table (no new merges learned,
  unlike `Train`) but reuses `Train`'s disk-backed intrusive-linked-list
  approach instead of `Encode`'s simple repeated merge-scan, so it scales
  to a full large corpus instead of just a short prompt. This is what
  would let `TokenCorpus` (TASK-010) be populated directly from a large
  corpus at training time, rather than the current slow per-call `Encode`
  path. Needs its own correctness tests (same output as `Encode` for
  inputs `Encode` can still handle, just faster) rather than assuming
  parity with `Train`'s different algorithm.
  Done: `BpeTokeniser.EncodeBulk` (`src/Tokeniser/BpeTokeniser.cs`) builds
  the same `LinkedTokenStream` `Train` uses, threads an intrusive
  occurrence chain per *known* merge pair, then applies merges in a
  single ascending pass over `_mergeRank` (no priority queue needed,
  since any pair a merge creates involves a token id that didn't exist
  until that merge ran, so it can only ever match a later merge).
  Returns a new `EncodedCorpus` (disk-backed via `MappedArray<int>`,
  over-allocated to the input's raw byte count as an upper bound).
  Deliberately driven off `_mergeRank` rather than the raw `_merges`
  list - found and fixed a real bug while writing the parity tests:
  `Train` can occasionally "learn" a merge for a pair that later reforms
  elsewhere in the corpus and gets merged again under a new id, silently
  overwriting that pair's `_mergeRank` entry and leaving the original
  merge's id permanently unreachable from raw bytes (`Encode` only ever
  consults `_mergeRank`, so it never produces that id either) - iterating
  `_merges` directly would have resurrected that dead id and diverged
  from `Encode`. Occurrences of the pair being merged are sorted into
  ascending position order before applying (the intrusive chain itself
  is LIFO) specifically to match `Encode`'s left-to-right, non-overlapping
  merge semantics for repeated-byte runs (e.g. "aaa" merges the first two
  bytes, not the last two). Verified via parity tests against `Encode`
  (multi-document corpora, an odd-length repeated-byte run, file-boundary
  non-merging, empty input, and `EncodedCorpus`'s out-of-range indexer).
  Depends on: TASK-002
  Required by: TASK-029

- [x] TASK-019: Optional disk-backed AdamW moment estimates - a
  constructor flag (e.g. `useDiskBackedState` + a scratch directory) on
  `AdamWOptimizer` backing `m`/`v` with `Tensor.ZerosOnDisk` instead of
  `Zeros`, mirroring the existing `MappedArray<T>` pattern. Deliberately
  lower priority than the other three: cheap to build when the time comes,
  but nothing built in this project so far has been large enough to need
  it - build this when a real model size makes it matter, not
  speculatively ahead of that.
  Done: `AdamWOptimizer` takes `useDiskBackedState` + `scratchDirectory`
  (throws `ArgumentException` if the flag is set without a directory).
  Disk-backed moment tensors are allocated once at construction and
  updated *in place* every `Step()` (`LoadInPlace` copies the freshly
  computed heap-backed values in, then disposes the transient) rather
  than replaced - replacing them the way the heap-backed default does
  would leak one scratch file per parameter per step, since nothing else
  would ever `Dispose` the one being replaced. `AdamWOptimizer` now
  implements `IDisposable` (a no-op for the heap-backed default, since
  `HeapFloatBuffer.Dispose` already is one) so disk-backed callers can
  release the moment tensors' mapped files once training finishes.
  Verified the disk-backed path is numerically identical to heap-backed
  (same closed-form first-step update, and a 200-step convergence test
  run specifically to catch a scratch-file leak - it would exhaust file
  descriptors well before 200 steps if moments were replaced instead of
  updated in place).
  Depends on: TASK-011

- [x] TASK-020: KV-cache for generation - the largest of these four, a
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
  Done: new `Model.GenerationCache` (plain `Tensor`, not `Variable` - a
  generation session never backpropagates) holds each layer's cached
  Key/Value; `MultiHeadAttention.ForwardIncremental` computes Q/K/V for
  only the *new* tokens, appends the new K/V onto the cache via a new
  `Tensor.Concat` (added since nothing like it existed - a plain,
  non-differentiable op, since the cache itself needs no gradient), and
  attends the new Q against the full resulting K/V.
  `ScaledDotProductAttention.Compute` gained a `queryOffset` parameter so
  its causal mask can be *rectangular* (new-query-rows-by-all-keys, not
  square) - query row i's real absolute position is `queryOffset + i`,
  not i, once Q and K cover different lengths; `PositionalEmbedding.Forward`
  gained an `offset` overload for the same reason (new tokens' absolute
  positions don't start at 0). `TransformerBlock.ForwardIncremental` and
  `GptModel.ForwardIncremental` thread the cache through; `FeedForward`/
  `LayerNorm` need no cache-aware version since they're already
  position-wise with no cross-sequence mixing. `Generation.TextGenerator`
  now uses the cache for every step after the first; a sliding-window
  step (context exceeding `MaxSequenceLength`) can't simply shift cached
  positions, so it calls `GenerationCache.Reset()` and rebuilds from the
  truncated window instead - the same one-step cost the old
  always-recompute approach paid on *every* step, now confined to the
  rare truncation steps. Correctness verified directly, per the task's
  own bar: `GenerationCacheTests` compares `ForwardIncremental`'s logits
  step-by-step against `Forward` recomputed from scratch on the same
  growing context (exact match, not "plausibly similar"), and
  `TextGeneratorTests` compares the public `GenerateTokenIds` API against
  a from-scratch reference reimplementation of the old always-recompute
  loop under greedy sampling (deterministic, so directly comparable).
  Solution-wide branch coverage held (`Model` 96.5%, `Generation` 96.4%).
  Depends on: TASK-007, TASK-008, TASK-009, TASK-013

- [x] TASK-021: CPU parallelism for `Tensor`'s hot ops - unlike TASK-015,
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
  Done: `Tensor.MatMul.cs`'s scalar and optimised paths both now spread
  independent output rows (a flattened batch*m index space, decomposed
  per row via a new pure `UnravelIndex` rather than the old
  incrementally-mutated `batchIdx` odometer, since parallel access needs
  a batch's coordinates computable directly from its flat index) across
  `Parallel.For` via a shared `ForEachRow` helper, once there are at
  least `MinRowsForParallelMatMul` (64) of them - below that, a plain
  sequential loop, since thread-scheduling overhead would dominate for
  e.g. the `[1]`-shaped scalar tensors used throughout this codebase for
  constants. The inner k-length reduction stays strictly sequential in
  both paths, preserving determinism (each row's output depends only on
  its own accumulation order, never on how many threads or what order
  ran). `MatMulOptimised` re-derives its `Span<float>`s from
  `IFloatBuffer.TryGetSpan` *inside* each row's closure rather than
  capturing them from the caller - `Span<T>` is a ref struct and can't be
  captured into a `Parallel.For` delegate, so this was a required change,
  not just a style choice; re-deriving is a cheap view over the same
  underlying array/pointer, not a copy. Elementwise ops left sequential
  (their O(n) cost is far smaller, as the task itself already flagged as
  lower priority). Verified via the existing MatMul test suite (all
  passing unchanged) plus two new tests at 100+/200+ rows (above the
  threshold): output matches an independent reference implementation,
  and repeated calls with the same input produce bit-identical results
  (proving the parallel path doesn't introduce nondeterminism). No
  automated performance assertion, per the task's own guidance - wall
  clock varies too much across environments to be a reliable test.
  Depends on: TASK-003

- [x] TASK-022: Modern regex-based pre-tokenisation for the BPE tokeniser -
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
  Done: new `Tokeniser.PreTokeniser.Split` (`src/Tokeniser/PreTokeniser.cs`)
  splits text via a `cl100k_base`-style pattern (contractions split from
  their stem, letter runs absorbing one leading non-letter/non-digit
  character, digit runs capped at 3, punctuation/symbol runs kept
  together, whitespace runs) using non-possessive quantifiers (a
  correctness-neutral portability simplification vs. the reference
  pattern's possessive ones - not worth the added complexity at this
  project's scale). `LinkedTokenStream.Build` (shared by `Train` and
  `EncodeBulk`) now treats each chunk as its own "document" the same way
  each whole file used to be, so chunk boundaries get the same -1
  Prev/Next sentinels file boundaries already relied on; `Encode` splits
  into chunks and merge-scans each independently, concatenating results.
  Found and fixed a real bug surfaced by the `EncodeBulk` parity tests
  while making this change: `LinkedTokenStream.Build`'s pre-existing
  `bytes.Length > 1` filter (meant to skip degenerate whole-file
  documents) was silently dropping every single-character chunk once
  "documents" became per-chunk rather than per-file - very common (any
  lone space or punctuation mark) and a real, not cosmetic, data-loss bug
  for `EncodeBulk`. Changed to `> 0` (skip only genuinely empty chunks).
  Tested: `PreTokeniser.Split` directly (canonical chunking examples, plus
  an exact-reconstruction property test across varied Unicode input -
  emoji, CJK, contractions, mixed whitespace, empty string); a
  chunk-boundary test confirming every chunk in a trained corpus still
  encodes/decodes to itself exactly (no merge crossing a boundary); and
  full encode-then-decode roundtrip tests through the new chunking.
  Solution-wide branch coverage held (measured per production assembly,
  per TASK-024's methodology - `Tokeniser` at 92.6%, `Program.cs` itself
  excluded per AGENTS.md).
  Depends on: TASK-001
  Required by: TASK-029

## Documentation
Added at the user's request.

- [x] TASK-023: Rewrite README.md as a lesson plan - walk through the whole
  process from a directory of raw `.txt` files all the way to a usable
  chatbot, one section per stage (mirroring PLAN.md's stage list), each
  covering: what problem that stage solves and what's actually happening
  conceptually, which source files to go read, and what to run to see it
  working (a CLI command where one exists, otherwise a code snippet or the
  relevant test project). Covers the full range including stages not built
  yet (TASK-014 chat CLI, TASK-016 instruction tuning) - be clear about
  what's runnable today versus what's planned, don't present unbuilt stages
  as if they already work.
  Done: rewritten as ten numbered stages mirroring PLAN.md's stage list
  exactly (tokeniser, tensor/autodiff, embeddings, attention/transformer
  block, model assembly, training loop, generation, chat CLI, optional
  `--optimised` backend, instruction tuning), each with what problem it
  solves, what's conceptually happening, source files to read, and a
  runnable command (a CLI invocation where one exists - tokeniser, chat -
  or the relevant test project/worked code snippet otherwise). By the
  time this task landed, every other planned task was already done, so
  unlike the task's original framing there were no not-yet-built stages
  left to caveat - the one honest caveat carried over instead: the chat
  CLI doesn't automatically apply the SFT prompt template, so a
  fine-tuned model needs prompts shaped to match it by hand. Project
  layout and status sections updated to match current reality (all ten
  stages plus the optimised backend, `Chat` project included, "known
  limitations" trimmed to what's still actually true - GPU/distributed
  training remains the one genuinely out-of-scope item).
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

## CLI gaps
Added at the user's request, while auditing the freshly-rewritten README
lesson plan (TASK-023): every stage that produces or consumes a model
artifact should have a CLI to actually do so, the same way the tokeniser
(TASK-001) and chat (TASK-014) do - pretraining and instruction tuning are
currently library-only, runnable only by pasting a C# snippet.

- [x] TASK-025: Pretraining CLI - a new console entry point (mirroring
  `src/Tokeniser`'s `TokeniserCli`/`Program.cs` split, testable via
  `Run(args, stdout, stderr)`) that runs TASK-012's `Trainer` end to end:
  load a trained tokeniser vocab (`BpeTokeniser.Load`), bulk-encode one or
  more corpus files into a `TokenCorpus` (TASK-018's `EncodeBulk`, not
  `Encode` - this is exactly the large-corpus case that task exists for),
  construct a fresh `GptModel` from CLI-supplied architecture flags
  (embedding dim, layer/head count, context length), train it for a given
  number of steps/batch size via `AdamWOptimizer/Trainer`, printing loss
  periodically (mirror `Trainer.Run`'s `onStep` callback), and
  `ModelCheckpoint.Save` the result to a CLI-supplied output path. Needs
  sensible flags/defaults for every hyperparameter the worked README
  example currently hardcodes (learning rate, embedding dim, layers,
  heads, context length, steps, batch size, target vocab size if training
  a tokeniser isn't assumed already done) plus a `--scratch-dir` the same
  way the tokeniser CLI requires one (real disk, not `tmpfs`) and,
  optionally, `--optimised` (TASK-015) to select the fast tensor backend
  for the run. Test the same way `TokeniserCli`/`ChatCli` are tested: a
  small fixture corpus, asserting the run completes and produces a loadable
  checkpoint whose loss actually dropped - not just "didn't crash."
  Once done, update README.md's stage 6 section to document the CLI (a
  runnable command, mirroring how stages 1/8 are documented) instead of
  only the current C#-snippet worked example - the lesson plan (TASK-023)
  should show the real command, not just point at library code.
  Done: `src/Pretrain/PretrainCli.cs` (+ `Program.cs`, same
  `Run(args, stdout, stderr)`-testable split as `TokeniserCli`/`ChatCli`).
  Loads a vocab via `BpeTokeniser.Load`, expands corpus file/directory
  positional args the same way `TokeniserCli` does (`*.txt` in a
  directory), bulk-encodes via `EncodeBulk`, and feeds the result straight
  into a **new** `TokenCorpus(EncodedCorpus, scratchDirectory)` constructor
  overload - added to `Training/TokenCorpus.cs` as part of this task,
  since it's exactly the resolution that class's own doc comment had been
  pointing at since TASK-018 landed (streams token ids from one
  disk-backed source into another, never materialising the whole corpus
  as a single in-memory collection, unlike the existing
  `IReadOnlyList<int>` constructor). Flags: `--embedding-dim`, `--layers`,
  `--heads`, `--context-length`, `--steps`, `--batch-size`,
  `--learning-rate`, `--weight-decay`, `--scratch-dir`, `--optimised`
  (TASK-015), all defaulting to the same values the README's worked
  example used to hardcode. Tested: usage/argument-error paths (mirroring
  `ChatCli`'s), missing vocab/corpus files, an empty corpus directory, the
  `--optimised` flag actually selecting `TensorBackend.Optimised`, and a
  genuine end-to-end run - not just "didn't crash" - asserting loss
  printed to stdout actually dropped and the saved checkpoint loads back
  with the requested architecture. `Pretrain` reached 100% branch
  coverage; solution-wide coverage held.
  Depends on: TASK-001, TASK-012, TASK-015, TASK-018
  Required by: TASK-026

- [x] TASK-026: Instruction-tuning (SFT) CLI - a new console entry point
  (same `Run(args, stdout, stderr)`-testable shape as TASK-025) that runs
  TASK-016's `SftTrainer` end to end: `ModelCheckpoint.Load` a *pretrained*
  checkpoint (produced by TASK-025's CLI, or the README's worked example),
  `BpeTokeniser.Load` its vocab, `SftDataset.Load` a JSON Lines
  instruction/response file, fine-tune for a given number of
  steps/batch size via `AdamWOptimizer/SftTrainer` (default learning rate
  should be visibly smaller than TASK-025's pretraining default - a tenth
  or less, per TASK-016's own guidance - not the same flag reused
  unchanged), and `ModelCheckpoint.Save` the result to a CLI-supplied
  *output* path that must differ from the input checkpoint path (refuse
  to overwrite the base pretrained model - TASK-016 was explicit that the
  base checkpoint is never overwritten, this CLI must not make that
  mistake easy to make by accident). Test the same way as TASK-025: a
  small fixture pretrained checkpoint + a small fixture SFT dataset,
  asserting the run completes, refuses an output path equal to the input
  path, and produces a loadable checkpoint whose masked loss actually
  dropped.
  Once done, update README.md's stage 10 section the same way TASK-025
  updates stage 6's: document the real command instead of only a
  C#-snippet sketch.
  Done: `src/Sft/SftCli.cs` (+ `Program.cs`, same `Run(args, stdout,
  stderr)`-testable split as `PretrainCli`/`ChatCli`). Refuses to run if
  `<output-checkpoint-path>` resolves (via `Path.GetFullPath`) to the same
  file as `<base-checkpoint-path>`, before loading anything - the base
  pretrained checkpoint is never at risk of being overwritten. Defaults
  `--learning-rate` to `3e-5` - a tenth of `PretrainCli`'s `3e-4` default,
  per TASK-016's own guidance, not the same flag value reused unchanged.
  Tested: usage/argument-error paths, the output/base-path-equality
  refusal, missing checkpoint/vocab/dataset files, a malformed dataset
  line, an empty dataset, `--optimised` actually selecting
  `TensorBackend.Optimised`, and a genuine end-to-end fine-tuning run
  asserting loss printed to stdout dropped and the resulting checkpoint
  loads back as a separate, correct file from the untouched base
  checkpoint. `Sft` reached 100% branch coverage; solution-wide coverage
  held.
  Depends on: TASK-016, TASK-025
  Required by: TASK-030

## Chat CLI improvements
Added at the user's request: even with TASK-016's instruction tuning and
TASK-026's SFT CLI, the chat CLI (TASK-014) still only approximates a real
instruction-tuned assistant experience - these two tasks close that gap at
the CLI/runtime level. Neither expands what data ships with the project:
the user/learner is still expected to supply their own pretraining corpus
and SFT dataset beyond the minimal `examples/sft-example.jsonl` starter, the
same as every other stage.

- [x] TASK-027: Instruction-tuned conversational mode for the chat CLI -
  today, `ChatCli` just appends each raw line of user input and each raw
  block of generated output to one growing token sequence, with no
  awareness that a fine-tuned model (TASK-016) actually expects
  `SftDataset`'s prompt template. Needs, opt-in (e.g. `--instruction-tuned`
  or similar, off by default so the existing raw-continuation behaviour -
  useful for a purely-pretrained checkpoint - isn't disturbed):
  - **Per-turn template wrapping.** Wrap each line of user input in the
    same template `SftDataset.Tokenize` uses (`"### Instruction:\n{0}\n\n### Response:\n"`)
    before encoding it, rather than encoding the bare line - reuse the
    template constant/logic from `SftDataset` rather than duplicating it
    (extract it to a small shared location if `SftDataset` doesn't already
    expose it in a reusable form).
  - **A stop condition.** Right now generation always runs to
    `--max-new-tokens` regardless of content. Needs a stop sequence (e.g.
    stop once generated text contains the next `"### Instruction:"` marker,
    trimming it back out of what's printed/kept as context) so a response
    doesn't run on into a hallucinated *next* user turn. `Generation.TextGenerator`/
    `TokenSampler` have no stop-sequence concept today - this is new
    surface area, not a flag on existing code.
  - **Template-consistent multi-turn history.** Each *prior* turn in the
    growing context should be shaped like the template too (its own
    `### Instruction:`/`### Response:` markers), not just the newest one -
    otherwise only the first turn matches what the model was tuned on and
    the conversation drifts out of distribution turn by turn.
  Test by comparing token sequences built by the CLI's new template path
  directly against what `SftDataset.Tokenize` would build for the same
  turns - they must match structurally, not just "look plausible" - plus a
  stop-sequence test proving generation actually halts at the boundary
  instead of running to `--max-new-tokens`.
  Depends on: TASK-014, TASK-016
  Required by: TASK-028

  **Done:** `SftDataset` now exposes `FormatPrompt(instruction)` and the
  public `InstructionMarker` constant so no caller duplicates the template
  string. `TextGenerator.GenerateTokenIdsUntilStopSequence` is new surface
  area: it generates one token at a time, decodes only the newly-generated
  tokens, and halts once that decoded text contains a stop sequence,
  re-encoding the trimmed text rather than slicing raw token ids (a BPE
  tokeniser's merges mean no fixed token-id sequence reliably spells a
  given string - only decoded text does). `ChatCli` gained `--instruction-tuned`:
  each turn's input is wrapped via `SftDataset.FormatPrompt` before
  encoding, generation stops at `SftDataset.InstructionMarker`, and because
  only the trimmed *response* text (never the marker) is appended back
  into the running context, every prior turn ends up shaped like the
  template automatically - no separate history-reformatting step needed.
  Tests: `SftDatasetTests` proves `FormatPrompt` produces token-identical
  output to `Tokenize`'s own prompt half; `TextGeneratorTests` proves the
  new method matches plain generation when no stop sequence appears, halts
  and trims correctly when one does (using an untrained byte-level
  tokeniser for fully deterministic decode), and preserves the prompt
  prefix; `ChatCliTests` proves the flag runs end-to-end and the banner
  text changes accordingly. Updated README stage 10 with real captured
  `--instruction-tuned` output, per this task's own note above.

- [x] TASK-028: Adjustable context window for the chat CLI - a CLI flag
  (e.g. `--context-length <n>`) capping how many of the *most recent*
  tokens a conversation keeps before the existing sliding-window
  truncation (TASK-013/TASK-020) kicks in, independent of just waiting for
  the model's own fixed `MaxSequenceLength` to be reached. Must not exceed
  the loaded model's `MaxSequenceLength` (validate and reject a value
  that's too large, don't silently clamp it). Useful for deliberately
  exercising the sliding-window/KV-cache-rebuild path on a model with a
  large max length without needing a conversation long enough to reach it
  naturally, and for controlling response latency/resource use on a slower
  machine without retraining. Test: a small `--context-length` truncates
  sooner than the model's real max would, a value exceeding the model's
  max is rejected with a clear error, and the default (no flag) behaves
  exactly as today (the model's own `MaxSequenceLength`).
  Depends on: TASK-014, TASK-020, TASK-027

  **Done:** `ChatCli` gained `--context-length <n>`, validated against the
  loaded model's `MaxSequenceLength` right after loading (a value that's
  too large is rejected with a clear error, never silently clamped).
  Defaults to the model's own `MaxSequenceLength` when omitted, so existing
  behaviour is unchanged by default. Applied via a small
  `TruncateToContextLength` helper called each turn before generation, on
  top of (not replacing) `TextGenerator`'s own `MaxSequenceLength` sliding
  window - this cap can only be equal to or tighter than that window, never
  looser, since it's validated against it. Tests: a small `--context-length`
  measurably changes greedy, deterministic output versus the default (proof
  it actually truncates sooner); a value exceeding `MaxSequenceLength` is
  rejected; passing the model's own `MaxSequenceLength` explicitly produces
  output identical to omitting the flag. Updated README stage 10 with real
  captured `--context-length` output, per this task's own note above.

## Scaling to a large real corpus / dataset

Flagged while answering a question about running a 250-book (~230 MB)
corpus through the pipeline: the "Memory and disk footprint" section of
README.md already documents, from real measurement, that peak RAM scales
roughly linearly with input text size (not just disk) - at 230 MB that
extrapolates to somewhere around 18 GB just for the bulk-encode/training
step, which is a genuine ceiling on this machine, not a footnote. Separately,
the SFT CLI's current `--steps`/`--batch-size` flags are a fine fit for the
6-example demo dataset but don't scale well to a dataset with hundreds or
thousands of examples.

- [x] TASK-029: Stream `LinkedTokenStream.Build`'s input instead of
  `File.ReadAllText` - the root cause identified in README.md's memory/disk
  footprint section: `BpeTokeniser.Train` and `EncodeBulk` both go through
  `LinkedTokenStream.Build`, which reads an entire input file into one
  heap-resident string via `File.ReadAllText` and only *then* hands derived
  data off to disk-backed (`MappedArray<T>`) storage - so peak RAM scales
  with input file size despite the disk-backed design elsewhere. Needs
  `PreTokeniser.Split`'s chunking (word/whitespace/punctuation-ish runs, a
  regex pattern - TASK-022) to work incrementally against a stream/reader
  in bounded-size blocks instead of one in-memory string, without letting a
  chunk (or a multi-byte UTF-8 character) get split across a block
  boundary - the read boundary must never become a new place merges can
  wrongly cross, the same invariant TASK-022 already protects at chunk
  boundaries. Test: peak managed heap usage (or at minimum, that no single
  allocation/string scales with input size) stays flat as input size grows
  across at least two genuinely different file sizes, while output
  (vocab/encoded tokens) stays byte-for-byte identical to today's
  `File.ReadAllText`-based result for the same input - this is a pure
  performance fix, not a behaviour change, so parity with existing
  encode/train output is the correctness bar. Update the README's memory/
  disk footprint section once done - the "honest surprise" paragraph
  describing this exact gap needs to say it's fixed, with fresh real
  measurements at the same 2/4/10 MB sizes (plus a larger size, e.g.
  100 MB+, to actually demonstrate flat RAM where the old numbers would
  have shown it climbing).
  Depends on: TASK-001, TASK-018, TASK-022

  **Done:** new `PreTokeniser.Split(TextReader reader, int bufferSize = 1 << 20)`
  reads bounded-size blocks and regex-matches `pending + block` each time;
  every match except the last found in a block is guaranteed complete (no
  later text can retroactively change where an already-terminated run
  ended - only the last match, if it's one of the pattern's unbounded-length
  alternatives, could still be continuing into the next block), so only
  that one gets held back as `pending` and re-scanned with the next block.
  `LinkedTokenStream.Build` now does two streaming passes per file - a
  first pass counting exact UTF-8 byte totals (`GetByteCount`, no
  allocation) to size the `MappedArray<T>`s precisely, a second pass
  actually encoding and filling them - instead of one `File.ReadAllText`
  pass. Twice the disk I/O, but peak heap no longer holds an entire file's
  text at once. Tested: `PreTokeniserTests` proves the streaming overload
  matches the in-memory one byte-for-byte across buffer sizes from 1 up to
  64 (including one deliberately smaller than a real word, to prove a
  chunk spanning many tiny blocks still comes back whole), plus an
  empty-reader case; full existing `BpeTokeniserTests`/`TokeniserCliTests`
  suites pass unchanged against the new streaming `Build`, proving output
  parity. Re-measured README.md's memory/disk footprint section at
  2/4/10/100 MB (`/usr/bin/time -v`, same methodology as before): peak RAM
  roughly halved at 10 MB (~840 MB → ~373 MB) and no longer scales anywhere
  near as steeply (a 100 MB corpus, which would have extrapolated to
  ~18 GB under the old code, now measures ~1.8 GB). Documented honestly
  that peak RAM still isn't perfectly flat - the disk-backed arrays'
  *write* working set still tracks corpus size (matching the existing
  16-bytes-per-input-byte scratch-estimate formula) - but those pages are
  OS-reclaimable under memory pressure, unlike the old unreclaimable heap
  string, which is what the disk-backed design was meant to buy in the
  first place. Solution-wide test suite: 396 passing; `Tokeniser` assembly
  at 95.7% branch coverage (`PreTokeniser` 100%, `BpeTokeniser` 98.8%).

- [x] TASK-030: Automatic epoch-based training for the SFT CLI - today
  `SftCli` takes a raw `--steps`/`--batch-size` pair with no relationship
  to dataset size, which the README's own stage 9 example works around
  manually (`--batch-size` set equal to the 6-example dataset size, so
  every example contributes to every step). That doesn't scale to a
  real dataset of hundreds or thousands of (instruction, response) pairs -
  the CLI should decide a sensible split into epochs (one full pass over
  the shuffled dataset) rather than making the user hand-compute
  steps-vs-batch-size themselves. Needs a new `--epochs <n>` flag (default
  a small number, e.g. 3, tuned against a real few-hundred-to-few-thousand-
  example dataset, not just the 6-example demo) that replaces `--steps` as
  the primary way to size a training run - each epoch is
  `ceil(datasetSize / batchSize)` steps over a shuffled pass, so total
  steps scale automatically with dataset size instead of being a fixed
  number the user must already know is "enough" for their data. Decide
  whether `--steps` stays as a lower-level escape hatch (e.g. mutually
  exclusive with `--epochs`, one or the other) or is retired in favour of
  epochs - needs to be resolved before implementation starts, not assumed.
  Shuffling between epochs (not just within `BatchSampler`'s existing
  window-drawing, which is pretraining's continuous-stream model, not
  SFT's example-list model) is new surface area for `SftTrainer`/
  `SftDataset`. Test: a dataset of a few hundred generated (instruction,
  response) pairs trains for the expected number of steps at a given
  `--epochs`/batch-size combination, loss trends down across epochs, and
  the existing 6-example demo continues to work with sensible defaults
  (not requiring the user to hand-tune `--batch-size` to the dataset size
  as README's stage 9 example currently has to).
  Depends on: TASK-016, TASK-026

  **Done:** resolved the open design question up front - `--steps` stays
  as a lower-level escape hatch (fixed count, sequential unshuffled order,
  no epoch concept), mutually exclusive with the new `--epochs` (an error
  if both are given). `SftTrainer` gained `RunEpochs(epochs, batchSize,
  random, onStep)`: each epoch Fisher-Yates-shuffles a fresh index
  permutation, then walks it in `batchSize`-sized slices (the final slice
  of an epoch is smaller when dataset size isn't a multiple of batch size -
  a new private `StepOn(exampleIndices)` helper, which both `Step` and
  `RunEpochs` now build on, averages by the batch's *actual* size rather
  than a fixed denominator so a short final batch's gradient scale is
  still correct). `SftCli` defaults to `--epochs 3` (tuned for a real
  dataset, not the tiny demo) with a `--batch-size` default of 8 -
  independent of dataset size, unlike the demo's old hand-tuned `6`.
  Updated README's stage 9 example to `--epochs 300` (no `--batch-size`
  override needed - the demo's 6 examples are already smaller than the
  default batch size of 8, so every epoch is naturally one full-batch
  step, reproducing the old manually-tuned convergence without the CLI
  ever needing to know the dataset size). Tested: `SftTrainerTests` proves
  `RunEpochs` invokes `onStep` exactly `ceil(datasetSize/batchSize) *
  epochs` times with correct (epoch, globalStep) pairs, handles a
  non-evenly-dividing batch size without error, and drives loss down
  substantially on a repetitive pattern; `SftCliTests` proves `--steps`
  and `--epochs` together is a clear error, `--epochs` prints the derived
  steps-per-epoch/total-steps, the no-flags default runs epoch-based
  training successfully, and a genuine end-to-end `--epochs` run produces
  a loadable checkpoint with dropped loss. All pre-existing `--steps`-based
  tests pass unchanged (steps stays a fully backward-compatible path).
  Solution-wide: 413 tests passing; `Sft` and `Training` both at 100%
  branch coverage.

## Optional GPU-accelerated backend (ILGPU)
Added at the user's explicit request specifically to demonstrate GPU-based
training as part of this project's lesson plan (PLAN.md stage 11) - not a
performance initiative for its own sake, and not previously planned. A
second explicit, opt-in library exception alongside TASK-015's Math.NET/
`System.Numerics.Tensors` CPU fast path: ILGPU JIT-compiles ordinary C#
into GPU kernels (CUDA/OpenCL/CPU accelerator), so this project still
writes and owns the kernel logic itself - the library changes *where* it
runs, not who wrote it. This machine has a discrete AMD GPU (Radeon
RX 6700 XT / Navi 22), so a genuine end-to-end demo here exercises ILGPU's
OpenCL path, not CUDA - the tasks below and any README documentation must
say so plainly rather than assume NVIDIA/CUDA by default. Tasks are
ordered as dependencies: wiring before a real kernel, a real kernel before
CLI/demo integration.

- [x] TASK-031: ILGPU dependency, context, and backend-selection plumbing -
  no GPU math yet, just the ability to select a GPU backend and have it do
  *something* correctly. Add the ILGPU NuGet package(s) to `Tensor` (or a
  new project, if keeping ILGPU's `Context`/`Accelerator` lifetime
  management out of `Tensor` itself turns out cleaner - a real design
  decision to make during implementation, not assumed here). Extend the
  existing `TensorBackend` enum (TASK-015: `Scalar`, `Optimised`) with a
  third value for GPU execution, selected the same process-wide way
  (`Tensor.Backend`, the existing `AsyncLocal<T>`-backed static) - no new
  selection mechanism. Needs accelerator *detection*, not an assumption a
  GPU exists: enumerate available ILGPU accelerators at startup, prefer a
  real GPU (CUDA or OpenCL) if present, and produce a clear, actionable
  error (not a crash) if the GPU backend is explicitly requested but no
  compatible accelerator is found - `Context`/`Accelerator` are disposable
  and process-lifetime, so decide and document exactly when they're
  created/torn down (once at first use? explicitly via a CLI flag at
  startup? both need to leave the process in a clean state either way).
  Test: accelerator detection/selection logic runs correctly against
  ILGPU's own CPU accelerator (always available, no real GPU hardware
  needed for this to be a meaningful, CI-safe test) and produces the
  documented clear error when a GPU is explicitly requested but
  unavailable - do not skip or hardcode-pass this case just because CI
  itself may lack a GPU.
  Depends on: TASK-015
  Required by: TASK-032

  **Done:** kept ILGPU's `Context`/`Accelerator` lifetime management
  inside `Tensor` itself (a new `GpuContext` static class) rather than a
  separate project - it's a peer of `TensorBackend`, not a distinct
  concern. `TensorBackend` gained `Gpu`. `GpuContext.GetAccelerator(bool
  allowCpuFallback = false)` lazily creates one process-wide `Context`
  (`builder.Default()`, which enables every accelerator kind ILGPU
  supports) and `Accelerator`, cached for reuse and torn down together via
  `Shutdown()` (mainly for test isolation, since the accelerator is a
  process-lifetime singleton otherwise). Real, on this specific dev
  machine: despite a discrete AMD GPU (Radeon RX 6700 XT) and an OpenCL
  ICD registration being present, the actual native OpenCL driver library
  the ICD points at isn't installed in this environment, so ILGPU only
  ever detects a CPU accelerator here - confirmed by direct probing before
  writing any code, not assumed. That's exactly the case
  `allowCpuFallback: false`'s error path exists for: rather than silently
  training "on GPU" while actually running on CPU, `GetAccelerator`
  refuses with a message naming every accelerator ILGPU actually found.
  The decision logic itself is extracted into a pure, public
  `ValidateAccelerator(AcceleratorType selected, bool allowCpuFallback,
  IEnumerable<AcceleratorType> available)` specifically so it's
  deterministically testable regardless of what hardware the test-running
  machine has - CI or this machine, with or without a working GPU driver,
  all exercise the exact same real code path. Tests: `GpuContextTests`
  proves the throw/no-throw decision for CPU-selected-without-fallback,
  CPU-selected-with-fallback, and Cuda/OpenCL-selected (both fallback
  settings) purely via `AcceleratorType` values (no hardware dependency);
  a hardware-touching test proves `GetAccelerator(allowCpuFallback: true)`
  returns a real, working `Accelerator` (allocates a device buffer, copies
  data to it, reads it back) on whatever accelerator is actually present;
  further tests prove caching (repeated calls return the same instance)
  and `Shutdown` (forces a fresh one next call). `Tensor` assembly:
  `GpuContext` at 100% branch coverage; solution-wide 421 tests passing.

- [x] TASK-032: A real GPU-accelerated matmul kernel via ILGPU - the actual
  math, building on TASK-031's plumbing. Matmul first, per TASK-015's own
  priority reasoning (O(n³) dominates a transformer forward/backward pass;
  elementwise ops are a stretch goal, not required for this to be worth
  having). Write the kernel as plain, readable C# (ILGPU compiles it, not
  a hand-rolled CUDA/OpenCL string) operating over ILGPU device-memory
  buffers - a genuinely new storage location alongside `IFloatBuffer`'s
  existing heap/disk-backed (`MappedArray<T>`) split, not a third case
  bolted onto either existing implementation; host↔device transfer is new
  surface area TASK-015's CPU-only fast path never had to consider.
  Mirrors TASK-015's scoping: most likely only applies to heap-backed
  tensors to start (a disk-backed tensor would need its own chunked
  transfer strategy to avoid pulling an entire large tensor onto the GPU
  at once - an open question to resolve during implementation, not
  pre-decided here, and out of scope for this task if it turns out to need
  its own design). Correctness bar, unchanged from TASK-015: the *same*
  test suite (including finite-difference gradient checks) must pass
  against this backend too, parametrised the same way TASK-015's tests
  were - not a separate, smaller GPU-only suite. Tests must default to
  ILGPU's CPU accelerator so they run without real GPU hardware; note
  clearly in the test file which tests would additionally benefit from
  being re-run manually against real GPU hardware (this machine's AMD
  card via OpenCL), since CI can't be assumed to have one.
  Depends on: TASK-031
  Required by: TASK-033

  **Done:** kept the scope decision explicit and separate from
  `GpuContext.GetAccelerator`'s own strictness: `Tensor.MatMulGpu` always
  calls `GetAccelerator(allowCpuFallback: true)` - its job is "compute the
  right answer on whatever accelerator is available," not "refuse if it's
  not a real GPU" (that stricter check belongs to a caller *choosing*
  `TensorBackend.Gpu` in the first place, i.e. TASK-033's CLI flag, not to
  every individual matmul call once it's already selected). New
  `Tensor.MatMul.Gpu.cs`: `MatMulGpu` mirrors `MatMulScalar`'s exact batch/
  broadcast math (same `MapBroadcastFlatIndex`-based per-batch offset
  computation) but launches one ILGPU kernel across every
  (batch*row, column) pair at once, rather than a CPU loop over rows - a
  kernel indexes strided device memory directly, so (unlike
  `MatMulOptimised`'s SIMD approach) no "transpose for a contiguous span"
  trick is needed. Device buffers for `a`/`b`/per-batch offsets/output are
  a genuinely new storage location, scoped (like TASK-015) to heap-backed
  operands only - falls back to `MatMulScalar` defensively if either isn't
  (`Zeros()` always is, in practice). The compiled kernel is cached per
  accelerator instance (`GetMatMulKernel`) since ILGPU compilation is real,
  non-trivial cost that shouldn't repeat every call.

  **Confirmed real, not assumed:** Coverlet's IL instrumentation is
  genuinely incompatible with ILGPU's kernel compilation - instrumenting
  `MatMulKernel`'s IL makes ILGPU's IR importer throw
  `InternalCompilerException`, even though the exact same code runs and
  passes correctly under plain `dotnet test` (no coverage collection).
  Root-caused (not worked around blindly) by running coverage collection
  in isolation, seeing every `Gpu`-backend test fail only under
  instrumentation, and confirming the same tests pass cleanly without it.
  Resolved by `[ExcludeFromCodeCoverage]` on `MatMulKernel` specifically
  (not the surrounding wrapper) - mirrors this project's existing
  precedent of excluding what a tool genuinely cannot measure (e.g.
  `Program.cs` composition roots), not a correctness or test-coverage
  compromise: the method is still fully exercised by real, passing tests.

  Existing `MatMul_*` (`TensorTests`) and `MatMul_*_GradientMatchesFiniteDifference`
  (`VariableTests`) theories gained a `TensorBackend.Gpu` case each,
  exactly the same test bodies TASK-015 already used for `Optimised` - no
  separate GPU-only suite. `Tensor.GpuContext` and `Tensor.Tests`/
  `Variable.Tests` share a named xUnit collection (`GpuContextTests`
  deliberately calls `GpuContext.Shutdown()`, which would otherwise race
  against a concurrently-running `Gpu`-backend matmul test in a different,
  parallel-by-default test class touching the same process-wide
  accelerator). On this specific dev machine, every `Gpu` case actually
  runs against ILGPU's CPU accelerator (see TASK-031's note on the missing
  OpenCL driver) - documented plainly in `TensorTests.cs` so this proves
  the kernel's math, not that it was verified against real GPU hardware;
  anyone with a working CUDA/OpenCL setup can re-run the same suite
  unchanged to additionally confirm that. Solution-wide: 431 tests
  passing; `Tensor.GpuContext` 100%, `Tensor.Tensor` 99.2% branch coverage.

- [ ] TASK-033: Wire the GPU backend into a CLI and demonstrate it end to
  end in README.md (a new stage 11 section, mirroring stage 9's
  `--optimised` treatment). Add a `--gpu` flag (name TBD - resolve any
  clash/overlap with `--optimised` before implementation: are `--gpu` and
  `--optimised` mutually exclusive, or does `--gpu` imply/require the
  optimised code path conceptually? decide, don't assume) to at least
  `Pretrain`'s CLI (the heaviest compute, most likely to show a measurable
  difference) - `Sft`/`Chat` are candidates too but not required if matmul
  volume there is too small to demonstrate anything. The README demo must
  be **honest about the actual win, not just "it ran"**: measure and
  report real wall-clock time for a comparable training run on the scalar
  path, TASK-015's optimised CPU path, and this GPU path, on this
  machine's real AMD GPU (OpenCL) - including the case where GPU transfer
  overhead makes a *tiny* toy-sized demo model slower on GPU than CPU,
  which is a real and likely honest finding worth documenting exactly the
  way the memory/disk footprint section's earlier "honest surprise"
  (TASK-029) was, not glossed over in favour of a flattering number.
  Update PLAN.md's "Known limitations / deferred" section once done to
  reflect single-GPU execution as delivered, not just planned.
  Depends on: TASK-025, TASK-032

## Notes
- Tasks are scoped for hand-rolled, no-library implementation per PLAN.md,
  except TASK-015 and TASK-031/032 (ILGPU), which are explicit,
  narrowly-scoped, opt-in exceptions.
- Work through tasks one at a time in order; ambiguities get clarified
  before implementation starts on that task, not up front for all of them.
