# AGENTS.md

Rules for anyone (human or AI agent) working in this repository. This file
is about *how* to work here; see [PLAN.md](PLAN.md) for the project roadmap
and [TASK.md](TASK.md) for granular, task-by-task status.

## What this project is

A personal learning tool: understand how LLMs actually work by building one
from first principles in C#/.NET, covering the whole path from a directory
of raw `.txt` files, through tokenisation, tensor math and automatic
differentiation, a trained transformer model, and out the other side as a
usable chatbot. Optimise every decision in this repo for *understanding*,
not for shipping a model quickly or squeezing out maximum performance.

## First principles, with one narrow exception

No ML/tokeniser/tensor libraries — no PyTorch, tiktoken, SentencePiece,
HuggingFace, ML.NET, Math.NET, `System.Numerics.Tensors`, etc. Tokenisation,
tensor math, autodiff, attention, and the training loop are written by hand
so every gradient and every mechanism is visible and understood, not a
black box behind a library call. This is deliberate: slower to reach a
working model, but understanding the mechanics is the entire point.

The one exception, and it is scoped narrowly: an **opt-in** fast path for
the tensor engine's hot ops (matmul above all), backed by exactly two named
libraries — Math.NET Numerics and `System.Numerics.Tensors` — selected by
an explicit flag, off by default. The hand-written scalar implementation
stays the real, default, always-correct reference; the fast path never
replaces it. The justification is narrow too: for the ops these two cover,
they're faster wrappers around math this project has already implemented
and understood from first principles, not new capability or a different
mechanism being hidden. No other library is in scope under this exception —
any other library dependency needs its own explicit conversation with the
project owner, not "well, we already made an exception once."

## Memory discipline: prefer disk, minimise heap

This is a standing constraint on every part of the codebase, not just the
tokeniser (which OOM-killed the development machine more than once before
this rule was written down — see commit history). Treat disk as scratch
space rather than assuming things fit in RAM:

- Prefer memory-mapped or streamed disk-backed storage over large in-memory
  arrays/collections wherever a structure scales with corpus size, dataset
  size, or model size: token streams, activations/gradients if they get
  large, checkpoints, batched training data, optimizer state.
- Minimise heap (anonymous) memory specifically. The OS can reclaim
  file-backed (disk-mapped) pages directly under memory pressure, but
  anonymous heap memory can only be reclaimed via swap — which is what
  actually causes an OOM kill. When choosing between "bigger disk-backed
  structure" and "bigger heap-allocated structure," prefer the former.
- Before adding a new large in-memory structure (a batch of activations, a
  full-corpus token buffer, an optimizer's moment estimates, a KV-cache),
  think through its memory scaling first and prefer a bounded/streamed/
  disk-backed approach over one that grows unboundedly with input size.
- See `src/Common/MappedArray.cs` for the established pattern (a
  memory-mapped scratch-file-backed array, shared by the tokeniser and the
  tensor engine) — reuse or adapt it rather than inventing a new mechanism.

## Use available CPU cores where it's safe to

Single-threaded by default across the whole codebase is wasteful on modern
multi-core hardware, and parallelising a loop is not the same kind of
concession as reaching for an ML library — .NET's Task Parallel Library
(`Parallel.For`/`Parallel.ForEach`, `System.Threading.Tasks`) is BCL, not a
new dependency, and it doesn't change the algorithm, only how many cores
execute the exact same code. Prefer it for independent, CPU-bound work that
scales with corpus/model/batch size (elementwise tensor ops, and especially
matmul's outer loops — the dominant cost in a forward/backward pass).

The one hard constraint: **never parallelise in a way that makes results
non-deterministic.** Parallelise independent outer loops (separate output
rows, batches, elements) freely; never parallelise a shared-state reduction
(e.g. the inner accumulation loop within a single matmul output element) -
that would make floating-point summation order, and therefore the exact
result, vary run to run, breaking reproducibility and any test that checks
results match an expected value. For small inputs, thread-scheduling
overhead can exceed the work being parallelised (e.g. the many `[1]`-shaped
scalar tensors used throughout this codebase for constants) — don't
parallelise unconditionally regardless of size.

This is a default to apply within whatever work is already in progress
(new code, or a task actively being implemented) — it's not a standing
invitation to go back and retrofit parallelism into already-working, tested
code outside of an assigned task (see the "don't optimise early" principle
below, and TASK-021 in TASK.md, which is specifically scoped to retrofit
this into the existing single-threaded `Tensor` engine).

## Testing: minimum 90% branch coverage

Every change must keep the codebase at or above 90% branch coverage,
project-wide. `coverlet.collector` is already referenced in every test
project; check coverage with:

```bash
dotnet test --collect:"XPlat Code Coverage"
```

Prefer tests that verify actual correctness over tests that merely execute
a code path for coverage's sake — a finite-difference gradient check or an
exact round-trip is worth more than an assertion-free smoke test, even
though both move the coverage number. Where a formula/algorithm has a
known correct value (e.g. uniform softmax input gives a known cross-entropy
loss), test against that value directly rather than only against another
implementation of the same logic.

## Code quality: SOLID and DRY, applied with judgement

Follow SOLID and DRY where they genuinely improve the code, not as a
checklist to satisfy regardless of fit. Concretely, in this codebase:

- **Don't optimise early.** Get something correct and tested first; only
  optimise once there's a real, measured reason to (a real corpus that's
  actually too slow, a real memory ceiling that's actually been hit — not
  a hypothetical one). Several tasks in TASK.md are explicitly deferred
  performance work for exactly this reason (e.g. TASK-019's disk-backed
  optimizer state is deliberately lower priority, to be built once a real
  model size makes it matter rather than speculatively ahead of that).
  This doesn't contradict "use available CPU cores" above — parallelising
  an already-correct loop over independent
  work is a cheap, safe default with no algorithmic or complexity cost
  once determinism is preserved, not the kind of speculative complexity
  ("maybe we'll need a cache/a new algorithm/a new abstraction") this rule
  is actually warning against.
- **Don't abstract early.** Add an interface, a base class, or a generic
  parameter when there are genuinely two+ concrete things that need it, not
  in anticipation of a hypothetical third. Duplication across two small,
  concrete implementations is often cheaper to live with than the wrong
  abstraction guessed at ahead of need.
- **Do reuse a working pattern once it exists.** `Common.MappedArray<T>`,
  `GaussianInit`, and the shared `GradientCheck` test helper are all cases
  where a second/third use case justified factoring something out — that's
  DRY applied honestly (after the duplication existed and was felt), not
  pre-emptively.
- **Document deliberate trade-offs where you make them**, the way PLAN.md's
  "Known limitations / deferred" section and inline code comments already
  do throughout this codebase — a documented, deliberate scope boundary is
  not the same thing as an oversight, but only if it's actually written
  down somewhere a future reader (human or agent) will find it.
