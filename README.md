# llmdev

A personal project to learn how LLMs work by building one from first
principles in C#/.NET — no ML, tokenisation, or tensor/autodiff libraries
(two narrow, explicitly-scoped exceptions, both opt-in and off by default,
neither a replacement for the hand-written implementation it speeds up: a
`--optimised` fast path for tensor math, stage 8 below; and a `--gpu` path
via ILGPU, stage 11 below, added specifically to demonstrate GPU-based
training). Every mechanism (tokeniser, tensor math, autodiff, attention,
training loop, generation, instruction tuning) is written by hand so it's
understood, not a black box behind a library call.

This README is a lesson plan: one section per stage, each covering what
problem the stage solves, what's actually happening conceptually, which
source files to go read, and what to run to see it working. Every stage
below is built and runnable today — see [Project status](#project-status)
for the honest caveat worth knowing before you expect too much of it as a
chatbot.

Stages are numbered in the order it makes sense to *approach* them, not the
order they were originally built in — see [PLAN.md](PLAN.md) if you want the
build history instead (its own stage numbers differ slightly, since
instruction tuning and the chat CLI were added after the original plan and
this README now presents them in a more sensible reading order: fine-tune
*before* you sit down to chat, since the chat CLI barely resembles a useful
assistant without it). [TASK.md](TASK.md) has the task-by-task history of
how everything below got built.

Two different kinds of section follow, and it matters which is which:

- **Stages with a CLI** (1, 6, 9, 10) each produce or use a model artifact
  (a trained vocabulary, a checkpoint) and are things you actually *run*.
  Every one of these includes a real example command and the real output it
  produced when this README was written.
- **Stages without a CLI** (2–5, 7, 8, 11) are learning steps: library code
  with no standalone command of its own, exercised only by tests and by the
  CLI stages above/after them (8 and 11 are flags on those CLIs, not
  learning steps in the usual sense, but still nothing you run standalone).
  Understanding what they do is what makes the CLI stages make sense, but
  there's nothing new to *run* for them beyond
  `dotnet test`.

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

## Stage 1 — Tokeniser

*(Has a CLI.)*

**Problem it solves:** a neural network works on numbers, not text. Something
has to turn raw text into a sequence of integers (and back again) before any
of the later stages can do anything.

**What's happening:** byte-level Byte-Pair Encoding (BPE) — the same idea
GPT-2/GPT-3-style tokenisers use. Start from a base vocabulary of the 256
possible byte values (so *any* input is representable, no "unknown token"
escape hatch needed), then repeatedly find the most frequent adjacent pair
of tokens in the training text and merge it into a new token, until a target
vocabulary size is reached. Text is first split into chunks (words, number
runs capped at 3 digits, punctuation runs, whitespace — modelled on
GPT-4/`cl100k_base`-style splitting rather than GPT-2's dated original
pattern) so a merge can never join, say, a trailing space into the next
word, or digits into an arbitrarily long number token.

**Source files:** `src/Tokeniser/BpeTokeniser.cs` (train, encode, decode,
save/load, plus `EncodeBulk` — a faster bulk-encode path for a large corpus
that reuses `Train`'s disk-backed data structures instead of `Encode`'s
simple repeated-scan approach), `src/Tokeniser/PreTokeniser.cs` (the
chunking regex), `src/Common/MappedArray.cs` (the disk-backed array type
`Train`'s internal state is built on, so a large corpus doesn't need to fit
in RAM).

**Run it:**

```bash
cd src/Tokeniser
dotnet run -- <vocab-size> <file-or-directory> [file-or-directory ...] [--scratch-dir <dir>]
```

Example, training a tiny 400-token vocabulary on a small repeated corpus
(the corpus used throughout this README's examples touches the same topics
as [`examples/sft-example.jsonl`](examples/sft-example.jsonl) below, so the
pipeline stays coherent end to end; real command, real output):

```text
$ dotnet run -- 400 big_corpus.txt --scratch-dir /mnt/data/scratch

Training BPE tokeniser on 1 file(s), 0.0 MB, target vocab size 400 (~0 MB disk scratch in /mnt/data/scratch)...
  vocab 300/400 (00:00)
  vocab 400/400 (00:00)
Trained vocabulary size: 400 in 00:00

Sample text:
The capital of France is Paris. Paris is a large city in France.
Red, blue, and yellow are the three primary colours. Blue is a colour.
A tokeniser converts text into a sequence of integer token ids a

Encoded (50 tokens):
393 399 308 310 266 307 46 307 266 256 314 264 331 398 274 310 260 392 44 368 44 300 377 347 363 365 390 121 382 46 357 266 256 304 260 65 383 366 354 397 362 256 374 395 308 361 332 277 349 256

Decoded (should match sample text):
The capital of France is Paris. Paris is a large city in France.
Red, blue, and yellow are the three primary colours. Blue is a colour.
A tokeniser converts text into a sequence of integer token ids a

Roundtrip match: True

Saved vocabulary + merges to vocab.bpe
```

This trains a vocabulary, prints a sample encode/decode roundtrip, and
writes it to `vocab.bpe`. Before training starts, estimated disk scratch
usage is checked against available disk space (and a `tmpfs`-backed scratch
directory, e.g. `/tmp` on many Linux distros, is refused — that's RAM under
a different name, defeating the point). Point `--scratch-dir` at real disk
if the default isn't.

```bash
cd tests/Tokeniser.Tests && dotnet test
```

## Stage 2 — Tensor + autodiff engine

*(No CLI — a learning/foundation step. Its logic runs inside every later
stage; there's nothing to invoke directly beyond its tests.)*

**Problem it solves:** everything from here on is matrix/vector math, and
training requires computing gradients of a loss with respect to every
learned parameter — by hand, for a model with millions of numbers, that's
infeasible.

**What's happening:** `Tensor` is an N-dimensional float array (shape,
strides, the core ops a transformer needs: add/multiply/divide, matmul,
transpose, reshape, reductions, softmax). `Variable` wraps a `Tensor` and
records every operation applied to it as a computation graph; calling
`Backward()` walks that graph in reverse, applying the chain rule at each
step, so a gradient never has to be derived and coded by hand for a new
operation — only its *forward* computation and its local derivative do.
Every op keeps this project's standing memory-discipline rule in mind:
storage is swappable between the managed heap and a disk-backed
`MappedArray<float>`-based buffer without changing the math above it, so a
tensor large enough to matter can live on disk instead of pushing the
process towards an OOM kill.

Understanding this stage is what makes stage 6's training loop (and its
CLI) make sense: every "forward pass, then backward pass, then optimizer
step" you'll see described later is built entirely out of the `Tensor`/
`Variable` operations defined here.

**Source files:** `src/Tensor/Tensor.*.cs` (the tensor type, split across
files by operation group), `src/Tensor/Variable*.cs` (the autodiff layer),
`src/Tensor/{HeapFloatBuffer,MappedFloatBuffer,IFloatBuffer}.cs` (the
storage abstraction).

**Run it:** no CLI - the tests *are* the way to see it working, including
finite-difference gradient checks (numerically perturb an input, compare
against the analytic gradient `Backward()` computed) rather than only
hand-derived expected values:

```bash
cd tests/Tensor.Tests && dotnet test
```

## Stage 3 — Embeddings

*(No CLI — a learning step. Its output feeds directly into stage 5's model
assembly.)*

**Problem it solves:** a token id is just an arbitrary integer with no
notion of meaning or similarity, and attention (stage 4) has no inherent
sense of sequence order — position 3 and position 30 look identical to it
without help.

**What's happening:** `TokenEmbedding` is a learned lookup table from token
id to a dense vector (trainable — gradients flow back into it during
training, so the vectors come to encode something useful about each token).
`PositionalEmbedding` is the same idea over *position* instead of token
identity, added elementwise to the token embedding so the model can tell
"first token" from "fifth token" apart.

**Source files:** `src/Model/TokenEmbedding.cs`, `src/Model/PositionalEmbedding.cs`.

**Run it:** no CLI - again a library piece, exercised by:

```bash
cd tests/Model.Tests && dotnet test
```

(covers embeddings, along with every other stage-4/5 component below — one
test project for the whole `Model` project.)

## Stage 4 — Attention + transformer block

*(No CLI — a learning step. Its output feeds directly into stage 5's model
assembly.)*

**Problem it solves:** a model needs a way to let each position in a
sequence gather information from *other* positions — "what came before this
that's relevant to predicting what comes next."

**What's happening:** scaled dot-product attention computes, for each
position, a weighted average of every position's "value" vector, where the
weights come from how well that position's "query" matches every position's
"key" (`softmax(Q @ K^T / sqrt(headDim)) @ V`). *Causal* masking (required
for a decoder-only, autoregressive model) prevents a position from
attending to positions after it — otherwise, during training, the model
would get to "see" the very token it's supposed to be predicting. Multi-head
attention runs several smaller attention computations in parallel (as a
batch dimension) so different heads can specialise in different kinds of
relationships. A full transformer block wraps attention with a
position-wise feed-forward network, layernorm (GPT-2's "pre-norm" placement
— normalise, then attend/feed-forward, then add the residual), and residual
connections, which are what let gradients flow through many stacked blocks
without vanishing.

**Source files:** `src/Model/ScaledDotProductAttention.cs`,
`src/Model/MultiHeadAttention.cs`, `src/Model/FeedForward.cs`,
`src/Model/LayerNorm.cs`, `src/Model/TransformerBlock.cs`.

**Run it:** no CLI - `cd tests/Model.Tests && dotnet test` (as above) —
includes an end-to-end check that causal masking actually holds through a
full block (changing a later token's value must never change an earlier
position's output).

## Stage 5 — Model assembly

*(No CLI — a learning step. This is the last piece before stage 6's CLI can
actually build and train something.)*

**Problem it solves:** stages 2–4 are all *pieces*; something has to stack
them into an actual language model.

**What's happening:** `GptModel` is a decoder-only, GPT-2-style model: token
+ positional embeddings, a stack of causal transformer blocks, a final
layernorm, and an output projection back to logits over the vocabulary. The
output projection reuses (rather than duplicates) the token embedding's
weight matrix, transposed — "weight tying", which halves what would
otherwise be two separate `[vocabSize, embeddingDim]`-sized matrices and is
a reasonable prior besides (a token's input representation and its output
"how likely is this next" score plausibly share structure). Stage 6's CLI
is what actually *constructs* one of these from architecture flags you
supply on the command line - this stage is where that architecture is
defined.

**Source files:** `src/Model/GptModel.cs`.

**Run it:** no CLI - `cd tests/Model.Tests && dotnet test` (as above) —
includes finite-difference gradient checks through the *entire* model, not
just individual layers.

## Stage 6 — Training loop

*(Has a CLI.)*

**Problem it solves:** a freshly constructed model's weights are random
noise. Training is the process of adjusting them, via gradient descent, so
the model actually gets better at predicting the next token.

**What's happening:** `TokenCorpus` holds a tokenised corpus as a flat,
disk-backed stream of token ids; `BatchSampler` draws fixed-length
(input, target) windows from it (the target is the input shifted one
position — "predict the next token"). `CrossEntropyLoss.Compute` scores how
well the model's predicted probability distribution matched the actual next
token at every position (with the standard numerically-stable
max-subtraction trick, since large logits would otherwise overflow `exp()`).
`SgdOptimizer`/`AdamWOptimizer` turn a computed gradient into a weight
update (AdamW optionally backs its per-parameter moment estimates with a
disk-backed tensor instead of the heap, for a large enough model).
`ModelCheckpoint` saves/loads a model's architecture + weights to/from a
single binary file. `Trainer` wires all of this into an actual loop:
forward pass, loss, backward pass, optimizer step, repeat.

**Source files:** `src/Training/TokenCorpus.cs`, `src/Training/BatchSampler.cs`,
`src/Training/CrossEntropyLoss.cs`, `src/Training/{SgdOptimizer,AdamWOptimizer}.cs`,
`src/Training/ModelCheckpoint.cs`, `src/Training/Trainer.cs`, `src/Pretrain/PretrainCli.cs`.

**Run it:**

```bash
cd src/Pretrain
dotnet run -- <vocab-path> <output-checkpoint-path> <corpus-file-or-directory> [file-or-directory ...]
  [--embedding-dim <n>] [--layers <n>] [--heads <n>] [--context-length <n>]
  [--steps <n>] [--batch-size <n>] [--learning-rate <f>] [--weight-decay <f>]
  [--scratch-dir <dir>] [--optimised]
```

Example, training a tiny model on the same corpus stage 1 tokenised, using
the `vocab.bpe` that stage produced (real command, real output — a toy-sized
model trained for only 300 steps, so the loss reaching ~0.01 reflects
memorising a small repetitive corpus, not general fluency):

```text
$ dotnet run -- vocab.bpe model.checkpoint big_corpus.txt \
    --embedding-dim 32 --layers 2 --heads 2 --context-length 128 \
    --steps 300 --batch-size 4 --learning-rate 0.003 --scratch-dir /mnt/data/scratch

Bulk-encoding 1 corpus file(s)...
Training a 2-layer, 32-dim GptModel for 300 steps (batch size 4, context length 128, 5220 tokens)...
step 0: loss 5.9995
step 100: loss 0.3141
step 200: loss 0.0303
step 299: loss 0.0121
Saved checkpoint to model.checkpoint.
```

Loads a vocabulary already trained via the [tokeniser CLI](#stage-1--tokeniser),
bulk-encodes the corpus (stage 1's `EncodeBulk`, not `Encode` — the
large-corpus path), trains a fresh `GptModel` from scratch, prints loss
periodically, and saves a checkpoint.

```bash
cd tests/Pretrain.Tests && dotnet test
cd tests/Training.Tests && dotnet test
```

both include genuine end-to-end proof: loss measurably drops over training
steps on a small, deliberately repetitive corpus.

## Stage 7 — Generation

*(No CLI of its own — its logic is what stage 10's chat CLI actually calls
at runtime. Understanding it explains what that CLI is doing under the
hood.)*

**Problem it solves:** a trained model outputs logits (raw, unnormalised
scores) over the vocabulary for "what comes next" — something has to turn
that into an actual chosen token, and do it repeatedly to produce more than
one token of output.

**What's happening:** `TokenSampler` turns a row of logits into a chosen
token id — greedy (always the highest-scoring token, deterministic),
temperature (rescale logits before sampling — lower is more conservative,
higher more random), and top-k/top-p (restrict sampling to only the most
likely candidates, by count or cumulative probability respectively).
`TextGenerator` runs this repeatedly: predict, sample, append, repeat — using
a KV-cache (`Model.GenerationCache`) so each step after the first only
computes the *new* token's contribution instead of recomputing every layer
over the whole growing context from scratch. Once the growing sequence
would exceed the model's context window, only the most recent tokens are
kept (a sliding window) — a KV-cache's positions can't simply be shifted
when that happens, so that one step rebuilds the cache from the truncated
window instead, the same one-off cost the simpler always-recompute approach
would have paid on *every* step.

**Source files:** `src/Generation/TokenSampler.cs`,
`src/Generation/SamplingOptions.cs`, `src/Generation/TextGenerator.cs`,
`src/Model/GenerationCache.cs`.

**Run it:** no CLI of its own - see stage 10's chat CLI below for the
hands-on way to exercise it, or the [worked example](#putting-it-together-training-and-generating-from-code)
further down for a direct code path.

```bash
cd tests/Generation.Tests && dotnet test
```

verifies sampling correctness (deterministic greedy, statistical
distribution checks for temperature/top-k/top-p) and, critically, that the
KV-cached generation path produces *exactly* the same output as a
from-scratch, non-cached recompute at every step — not just "plausibly
similar text."

## Stage 8 — Optional optimised math backend

*(No CLI of its own — it's a flag (`--optimised`) accepted by the stage 6,
9, and 10 CLIs, not a pipeline stage you run on its own.)*

**Problem it solves:** the hand-written scalar tensor math (stage 2) is
correct and easy to reason about, but slow — useful for learning, painful
once you actually want to wait for a training run to finish.

**What's happening:** an opt-in fast path for `Tensor.MatMul` (the dominant
cost in a transformer forward/backward pass), backed by
`System.Numerics.Tensors.TensorPrimitives` — this project's one deliberate,
narrowly-scoped exception to "no ML/tensor libraries," justified because
it's a faster *wrapper* around math already implemented and understood from
first principles in stage 2, not new/different mechanics being hidden
behind a library call. Off by default; the hand-written scalar
implementation remains the always-correct reference, never replaced.
Independent output rows are also spread across CPU cores via
`Parallel.For` (both the scalar and optimised paths), once there are enough
of them to be worth the thread-scheduling overhead — plain .NET TPL, not a
new dependency, and a size threshold below which everything just runs on
one thread as before. A disk-backed tensor always declines the optimised
path and falls back to scalar automatically, keeping the memory-discipline
rule intact no matter which backend is selected.

**Source files:** `src/Tensor/TensorBackend.cs`, `src/Tensor/Tensor.MatMul.cs`,
`src/Tensor/{HeapFloatBuffer,MappedFloatBuffer}.cs` (`TryGetSpan`).

**Run it:** no CLI of its own - pass `--optimised` to the [pretraining](#stage-6--training-loop),
[instruction-tuning](#stage-9--instruction-tuning-sft), or [chat](#stage-10--interactive-chat-cli)
CLIs, or set `Tensor.Backend = TensorBackend.Optimised` in your own code
before calling into the model. Verified by running the *same* test suite
(including every stage-2 finite-difference gradient check) against both
backends and confirming equivalent results — see the `[Theory]`-parametrised
tests in `tests/Tensor.Tests/{TensorTests,VariableTests}.cs`.

## Stage 9 — Instruction tuning (SFT)

*(Has a CLI. Deliberately placed before the chat CLI below - fine-tuning is
what makes the chat CLI worth using; without it you're chatting with a raw
next-token predictor.)*

**Problem it solves:** a model trained purely on raw text (stages 1–7)
continues text in the style of its training corpus — it has no particular
tendency to *answer questions* or *follow instructions* unless that corpus
happened to already look like dialogue. Instruction tuning (supervised
fine-tuning, SFT) continues training a pretrained checkpoint on
(instruction, response) example pairs instead, so the model gets some
actual tendency to behave like something worth having a conversation with.

**What's happening:** two things stage 6 didn't need. First, an
example-based dataset — `SftDataset` loads JSON Lines
(`{"instruction": "...", "response": "..."}` per line), wraps each
instruction in a fixed prompt template, and tokenises the templated prompt
and the response *separately* so the response always starts on an exact
token boundary — distinct from stage 6's `TokenCorpus`/`BatchSampler`, which
draw sliding windows from one continuous stream with no concept of an
"example" at all. Second, masked cross-entropy loss
(`CrossEntropyLoss.ComputeMasked`) — the loss is averaged only over
*response* token positions, not the instruction/prompt tokens, since a
model shouldn't be penalised for not "predicting" words that were handed to
it as the question. `SftTrainer` wires a pretrained model (loaded via
`ModelCheckpoint.Load`, never overwritten — save the fine-tuned result to a
separate checkpoint file) together with this dataset and loss, the same way
`Trainer` does for pretraining, using a smaller learning rate (a tenth or
less of pretraining's is a common starting point).

**Where the (instruction, response) pairs themselves come from** is a
data-preparation question, separate from this project's "no libraries" rule
(that rule is about runtime code dependencies, not how a data file gets
written offline). In practice, some combination of: manually authoring a
small, high-quality, deliberately-scoped set; reformatting content you
already have the rights to (docs, FAQs, support transcripts); generating
draft pairs with a separate, stronger model used purely as an offline,
one-time data-prep tool (review before use, don't accept uncritically); or
adapting an existing public instruction-tuning dataset (mind licensing,
and reformat to this project's template).

**Example dataset file:** [`examples/sft-example.jsonl`](examples/sft-example.jsonl)
is a small, ready-to-use starter file - one JSON object per line:

```json
{"instruction": "What is the capital of France?", "response": "The capital of France is Paris."}
{"instruction": "Name three primary colours.", "response": "Red, blue, and yellow are the three primary colours."}
```

**Source files:** `src/Training/{SftExample,SftDataset,SftTrainer}.cs`,
`src/Training/CrossEntropyLoss.cs` (`ComputeMasked`), `src/Sft/SftCli.cs`.

**Run it:**

```bash
cd src/Sft
dotnet run -- <base-checkpoint-path> <vocab-path> <dataset-path> <output-checkpoint-path>
  [--epochs <n> | --steps <n>] [--batch-size <n>] [--learning-rate <f>] [--weight-decay <f>] [--optimised]
```

**`--epochs`** (default 3, TASK-030) is the primary way to size a training
run: each epoch is one full, freshly-shuffled pass over the dataset
(`ceil(datasetSize / batchSize)` steps), so the total amount of training
scales with dataset size automatically instead of you having to
hand-compute a step count against a dataset you may not have measured yet.
`--steps` still exists as a lower-level escape hatch - a fixed step count,
sequential (unshuffled) example order, no notion of an epoch - but the two
are mutually exclusive; specifying both is an error. `--batch-size`
defaults to 8, a fixed number independent of dataset size (unlike the
demo below, which used to need `--batch-size` hand-set to match its
6-example dataset exactly).

Example, fine-tuning the checkpoint stage 6 produced above on
[`examples/sft-example.jsonl`](examples/sft-example.jsonl) (real command,
real output). The 6-example demo dataset is far smaller than
`--batch-size`'s default of 8, so every epoch here is one full-batch step
over all 6 examples - `--epochs 300` reproduces the same "every example,
every step" convergence the demo used to get by manually setting
`--batch-size` to the dataset size, without the CLI needing to know the
dataset size at all:

```text
$ dotnet run -- model.checkpoint vocab.bpe examples/sft-example.jsonl model-sft.checkpoint \
    --epochs 300 --learning-rate 0.0001

Fine-tuning on 6 example(s) for 300 epoch(s) (1 step(s)/epoch, 300 step(s) total, batch size 8)...
step 0: loss 3.9941
step 100: loss 1.1884
step 200: loss 0.7917
step 299: loss 0.6425
Saved fine-tuned checkpoint to model-sft.checkpoint.
```

The output checkpoint path must differ from the base checkpoint path - the
CLI refuses to overwrite the base pretrained checkpoint. `--learning-rate`
defaults to a tenth of stage 6's pretraining default, per this stage's own
guidance above.

```bash
cd tests/Sft.Tests && dotnet test
cd tests/Training.Tests && dotnet test
```

includes an end-to-end proof that loss actually drops on a small repetitive
instruction/response pattern, mirroring stage 6's pretraining equivalent,
for both `--epochs` and `--steps`.

## Stage 10 — Interactive chat CLI

*(Has a CLI.)*

**Problem it solves:** everything before this point is only usable via a
one-off command or C# snippet — there's no way to just sit down and talk to
a trained model turn-by-turn.

**What's happening:** loads a saved `ModelCheckpoint` + tokeniser vocab,
then loops: read a line of input, generate a continuation, print it,
repeat. Conversation state is a single growing token-id sequence (not
re-encoded text each turn), so multi-turn context accumulates correctly.

**Honest expectation-setting, not a bug:** a model that only went through
stage 6 (pretraining) is a *raw next-token-prediction model*, not an
instruction-tuned assistant - it will continue text in the style of
whatever it was trained on, not necessarily answer questions or follow
instructions. Stage 9's fine-tuning makes it meaningfully more likely to
behave like something worth talking to - but by default, this CLI still
reads one line of input as one turn with no special handling for stage 9's
multi-line prompt template. That's what `--instruction-tuned` below is
for.

**Source files:** `src/Chat/ChatCli.cs`, `src/Generation/TextGenerator.cs`
(`GenerateTokenIdsUntilStopSequence`).

**Run it:**

```bash
cd src/Chat
dotnet run -- <checkpoint-path> <vocab-path> [--temperature <f>] [--top-k <n>] [--top-p <f>]
  [--max-new-tokens <n>] [--context-length <n>] [--instruction-tuned] [--optimised]
```

**Default mode** - reproduces the `"### Instruction:\n{instruction}\n\n### Response:\n"`
shape the model was actually fine-tuned on. Example, chatting with the
stage-9 fine-tuned checkpoint from above (real command, real output -
greedy sampling, and a *tiny*, undertrained toy model by design, so don't
expect fluent prose; this is here to show the mechanism working end to
end, not to demonstrate quality). Typing a bare one-line question doesn't
reproduce the template shape the model was fine-tuned on, so instead of
answering it continues into whatever text its training distribution makes
most likely next:

```text
$ dotnet run -- model-sft.checkpoint vocab.bpe --temperature 0 --max-new-tokens 15

Loaded model and vocabulary. This is a raw next-token-prediction model, not an
instruction-tuned assistant: it will continue text in the style of whatever it was
trained on, not necessarily answer questions or follow instructions.
Type /exit (or send EOF, e.g. Ctrl+D) to leave.

> What is the capital of France?
s freat tocabu     av
> /exit
Goodbye.
```

**`--instruction-tuned`** (TASK-027) - opt-in, off by default so a purely
pretrained checkpoint's existing raw-continuation behaviour isn't
disturbed. Each turn's input is wrapped via `SftDataset.FormatPrompt`
before encoding (the exact same template `SftDataset.Tokenize` uses, not a
reimplementation of it), and generation halts at the next
`SftDataset.InstructionMarker` (`"### Instruction:"`) instead of running on
to `--max-new-tokens` regardless of content - new surface area in
`TextGenerator.GenerateTokenIdsUntilStopSequence`, since decoded *text* is
what's checked for the marker (a BPE tokeniser's merges mean no fixed
token-id sequence reliably spells a given string). Because only the
trimmed response text - never the marker itself - gets appended back into
the running conversation, every prior turn ends up shaped like the
template automatically, without a separate history-reformatting step:

```text
$ dotnet run -- model-sft.checkpoint vocab.bpe --instruction-tuned --temperature 0 --max-new-tokens 30

Loaded model and vocabulary. Instruction-tuned mode: each turn is wrapped in the same
prompt template the model was fine-tuned on (TASK-016), and generation stops at the next
turn boundary instead of running on.
Type /exit (or send EOF, e.g. Ctrl+D) to leave.

> What is the capital of France?
The capital of France is Paris. Paris is a large city in France.
Red, blue, and yellow are the thre
> /exit
Goodbye.
```

Honest note on that transcript: this particular tiny, greedy, undertrained
demo model answers the question correctly, then keeps going instead of
actually halting at a `### Instruction:` boundary within 30 tokens - it
never learned a strong enough tendency to reproduce that exact marker at
this toy scale (pushed further, greedy decoding on a model this small and
this overfit tends to degenerate into repetition loops rather than
reproduce it either). The halting mechanism itself is proven correct by
`Generation.Tests`' dedicated stop-sequence tests (a case engineered so
the marker genuinely appears in generated text, confirming generation
halts and trims at exactly that point) - this demo shows the *template
application* working correctly, not a guarantee that any given toy model
will spontaneously hit the boundary in any given number of tokens.

**`--context-length <n>`** (TASK-028) - caps how many of the most recent
tokens a conversation keeps before `TextGenerator`'s own sliding window
(governed by the model's `MaxSequenceLength`) would otherwise kick in.
Validated against the loaded model's `MaxSequenceLength` - a value that's
too large is rejected outright, never silently clamped:

```text
$ dotnet run -- model-sft.checkpoint vocab.bpe --context-length 9999

--context-length 9999 exceeds this model's MaxSequenceLength (128).
```

A small value truncates conversation history far sooner than the model's
own 128-token limit would, which shows up as visibly degraded output once
too little context remains (real command, real output, three turns in a
row with `--context-length 16`):

```text
$ dotnet run -- model-sft.checkpoint vocab.bpe --context-length 16 --temperature 0 --max-new-tokens 10

> What is the capital of France?
s freat tocabu  
> Name three primary colours.
.. Paris is a tocabu
> What does BPE stand for?
s for Byte-Pair En
> /exit
Goodbye.
```

Omitting the flag behaves exactly as before - equivalent to passing the
model's own `MaxSequenceLength` explicitly.

```bash
cd tests/Chat.Tests && dotnet test
cd tests/Generation.Tests && dotnet test
```

## Stage 11 — Optional GPU-accelerated backend (ILGPU)

*(No CLI of its own — a flag (`--gpu`) accepted by the stage 6 CLI, the
same way stage 8's `--optimised` is, not a pipeline stage you run on its
own. Added after the original plan, at the user's explicit request
specifically to demonstrate GPU-based training as part of this lesson
plan.)*

**Problem it solves:** stages 2 and 8 only ever run on the CPU - the
hand-written scalar path, or stage 8's SIMD-accelerated one. Neither can
demonstrate what GPU-based training actually looks like.

**What's happening:** a third, opt-in `Tensor.MatMul` backend
(`TensorBackend.Gpu`), backed by [ILGPU](https://github.com/m4rs-mt/ILGPU) -
this project's second and only other deliberate library exception, alongside
stage 8's. ILGPU JIT-compiles ordinary C# into a real GPU kernel (CUDA,
OpenCL, or its own CPU accelerator) at runtime, so - the same justification
as stage 8's - this project still writes and owns the actual matmul kernel
(`Tensor.MatMulGpu`/`MatMulKernel` in `src/Tensor/Tensor.MatMul.Gpu.cs`)
itself; the library changes *where* it runs, not who wrote it. `GpuContext`
(`src/Tensor/GpuContext.cs`) manages one process-wide ILGPU `Context`/
`Accelerator`, created lazily and reused. Selecting `--gpu` on the
pretraining CLI runs a *preflight* check before any training starts: by
default it requires a genuine CUDA or OpenCL accelerator and refuses with a
clear error if none is found, rather than silently training on ILGPU's own
CPU accelerator while claiming to demonstrate GPU execution - pass
`--gpu-allow-cpu-fallback` to accept that CPU accelerator anyway (real
command, real output, both below). `--gpu` and `--optimised` select
different backends and can't be combined.

**Honest finding, measured on this machine, not assumed - and revised once
already, exactly because of what "measured, not assumed" actually
requires:** this project's dev machine has a discrete AMD GPU (Radeon
RX 6750 XT / Navi 22 - `gfx1030`). The first time this section was written,
that GPU genuinely wasn't reachable: an OpenCL ICD registration existed,
but the native runtime library it pointed at wasn't installed, so ILGPU
only ever found a CPU accelerator. Root-caused (not worked around) down to
one specific missing package: `clinfo` could already see the GPU via
`libOpenCL.so.1`, but .NET's native-library probing for ILGPU's OpenCL
P/Invoke layer needed the unversioned `libOpenCL.so` symlink, which only
the `-devel` package (`ocl-icd-devel`) provides on this Fedora machine -
installing it (`sudo dnf install ocl-icd-devel`) fixed detection
immediately, with no code change. Real GPU execution below is the result
of that fix, re-measured afterwards - the numbers in the first version of
this section (GPU slower than either CPU path, via ILGPU's CPU accelerator)
are superseded, not still true.

**Source files:** `src/Tensor/GpuContext.cs`, `src/Tensor/Tensor.MatMul.Gpu.cs`,
`src/Tensor/TensorBackend.cs`, `src/Pretrain/PretrainCli.cs`.

**Run it:** no CLI of its own - pass `--gpu` to the
[pretraining CLI](#stage-6--training-loop) (`--gpu-allow-cpu-fallback` also
exists, for a machine without a working GPU driver - see the note above).
Real command, real output, genuinely running on the AMD GPU via OpenCL this
time (`GpuContext`'s own accelerator-selection logic, unchanged since
TASK-031, already preferred a real GPU whenever `Context.Devices` reported
one - it just had nothing to prefer until the driver was fixed):

```text
$ dotnet run -- vocab.bpe model.checkpoint big_corpus.txt \
    --embedding-dim 64 --layers 2 --heads 2 --context-length 64 \
    --steps 30 --batch-size 4 --learning-rate 0.0003 --gpu

Bulk-encoding 1 corpus file(s)...
Training a 2-layer, 64-dim GptModel for 30 steps (batch size 4, context length 64, 3840 tokens)...
step 0: loss 6.1716
step 29: loss 5.1018
Saved checkpoint to model.checkpoint.
```

Wall-clock comparison, same corpus/model/step count throughout (a small
2-layer, 64-dim model, 30 steps, batch size 4, context length 64 - toy-sized
by design, same as every other worked example in this README):

| Backend | Flag | Wall-clock (`real`, `time`) |
|---|---|---|
| Scalar (default) | *(none)* | 25.2s |
| Optimised (stage 8) | `--optimised` | 23.5s |
| GPU (real AMD RX 6750 XT, OpenCL) | `--gpu` | 23.2s |

**The honest reading of that table:** real GPU execution here is roughly
on par with both CPU paths - not the dramatic win a GPU's reputation might
suggest, and not the earlier "GPU is slower" finding either. At this
toy scale (a 64-dim, 2-layer model), per-matmul kernel-launch and
host↔device transfer overhead is still large relative to the actual
compute each matmul does, so a real GPU's parallelism advantage doesn't
get much room to show up - the same fundamental limitation the earlier,
CPU-accelerator-only measurement demonstrated more starkly. A genuinely
bigger model (wide enough that each matmul's own compute dominates its
launch/transfer overhead) would very likely show GPU pulling ahead of both
CPU paths - consistent with why GPUs are the standard choice for real LLM
training - but confirming that would need a much longer training run than
this README's toy-sized, fast-to-reproduce examples are meant for (a
256-dim/4-layer attempt during this write-up didn't finish in several
minutes on the scalar path alone, only underlining the point). Reporting
"roughly even at toy scale, real advantage would need a bigger model to
confirm" plainly, rather than only showing a number and calling it a GPU
win, is the same commitment to honest measurement this README's memory/disk
footprint section makes.

**Does a bigger *corpus* change that, even with the same tiny model?**
Measured directly rather than assumed: same 2-layer, 64-dim model, same
fixed 30 steps, against real 10 MB, 20 MB, and 50 MB corpora (the same
sizes this README's memory/disk footprint section already uses, for a
consistent point of reference):

| Corpus | Optimised (`--optimised`) | GPU (`--gpu`) |
|---|---|---|
| 10 MB | 26.2s | 25.1s |
| 20 MB | 30.6s | 29.1s |
| 50 MB | 42.1s–44.7s (2 runs) | 43.3s |

Both backends grow together as corpus size grows - because the step count
is fixed regardless of corpus size, that growth is almost entirely
bulk-encoding time (backend-independent; matches the tokeniser's own
linear-with-input-size behaviour from the footprint section), not the
matmul-heavy training loop itself. The two backends stay within noise of
each other at every size, consistent with the single-corpus finding above:
a bigger *corpus* doesn't change the picture, since it doesn't add more
per-step matmul work - only a bigger *model* would. (One 50 MB GPU run
initially read 65.2s, a clear outlier against a repeat run at 43.3s and
against the pattern at every other size - re-run and reported here rather
than left in, since a number that doesn't reproduce isn't a finding, it's
noise.)

**Source files** (tests): `tests/Tensor.Tests/{GpuContextTests,TensorTests,VariableTests}.cs`,
`tests/Pretrain.Tests/PretrainCliTests.cs`.

```bash
cd tests/Tensor.Tests && dotnet test
cd tests/Pretrain.Tests && dotnet test
```

verified by running the *same* test suite (including every gradient check)
against the `Gpu` backend too, the same way stage 8's `--optimised` is
verified - see the `[Theory]`-parametrised tests in `TensorTests.cs`/
`VariableTests.cs`. These now genuinely run against the real AMD GPU via
OpenCL on this machine, not ILGPU's CPU accelerator - both prove the
kernel's math is correct; only which hardware executed it changed.

**Follow-up: does keeping model weights resident on the GPU help?**
(TASK-034/035/036) Once every matmul call stopped re-uploading an
already-device-resident operand (TASK-035), the natural next question was
whether keeping a model's *weights* GPU-resident for a whole training run
- via new `Tensor.MoveToGpuInPlace`/`MoveToHostInPlace` (TASK-034) and a
new `--gpu-resident-weights` flag (requires `--gpu`) - actually helps.
**Measured, not assumed: it's dramatically worse, not better.** Same tiny
model (2-layer, 32-dim, 5 steps, batch size 2, context length 32) throughout:

| Backend | Wall-clock (`real`, `time`) |
|---|---|
| Scalar (default) | 3.5s |
| Optimised (stage 8) | 3.8s |
| GPU (`--gpu`) | 4.1s |
| GPU + resident weights (`--gpu --gpu-resident-weights`) | 153.7s (≈ 37× slower) |

**Why:** TASK-035 deliberately only taught `Tensor.MatMulGpu` to use an
already-resident operand directly - it left every *other* op (backward's
`Transpose`, the elementwise ops behind it, `AdamWOptimizer`'s
`SubtractInPlace` weight update) untouched, and none of them have a
device-resident code path. `GpuFloatBuffer`'s indexer is correct against
any of them, but it's a genuine host↔device round trip *per element*, not
a bulk transfer - so once weights stay resident, every backward pass and
every optimizer step touches every parameter element-by-element instead of
in one batch. `--gpu-resident-weights` is opt-in and off by default
specifically because of this - it exists to make the mechanism (and this
exact finding) demonstrable, not because it's recommended. Closing this
gap for real would mean giving `Transpose`, the elementwise ops, and the
optimizer's update their own device-resident code paths too (real,
substantial further work, not attempted here) - reporting "this made
things much worse, and here's precisely why" is exactly the kind of
finding this README's honesty commitment exists for, the same as the
memory/disk footprint section's own surprise.

```bash
cd tests/Tensor.Tests && dotnet test
cd tests/Pretrain.Tests && dotnet test
```

`TensorTests` covers `MoveToGpuInPlace`/`MoveToHostInPlace` directly
(value round-tripping, no-op cases, same-object identity, and a `MatMul`
producing the same result whichever operand was moved) and
`MatMul`-with-a-resident-operand correctness (either operand, both, and
the same resident operand reused correctly across several calls);
`PretrainCliTests` proves `--gpu-resident-weights` requires `--gpu` and
trains correctly (loss values are finite, a checkpoint is produced) at a
deliberately tiny fixture size, given how slow this path is.

## Putting it together: training and generating from code

Every stage that touches a model artifact now has a CLI (tokeniser -
stage 1, pretraining - stage 6, instruction tuning - stage 9, chat -
stage 10). The example below walks through stages 2–7 (pretraining +
generation) as a from-scratch C# snippet anyway, for anyone who wants to
see the pieces wired together directly rather than through the stage 6
CLI; stage 9's fine-tuning loop looks the same shape, just with
`SftDataset.Load` + `SftTrainer` in place of `TokenCorpus`/`BatchSampler` +
`Trainer` (or just use [its CLI](#stage-9--instruction-tuning-sft)
directly on a checkpoint this example produced).

```csharp
using Model;
using Tokeniser;
using Training;
using Generation;

const string ScratchDirectory = "/home/you/.cache/llmdev-scratch"; // real disk, not /tmp - see stage 1 above
const string CorpusPath = "corpus.txt";

// Train (or load) a tokeniser, then encode the training corpus into a
// disk-backed token stream and a batch sampler over it.
var tokeniser = new BpeTokeniser();
tokeniser.Train([CorpusPath], targetVocabSize: 2000, ScratchDirectory);
tokeniser.Save("vocab.bpe");

var tokenIds = tokeniser.Encode(File.ReadAllText(CorpusPath));
using var corpus = new TokenCorpus(tokenIds, ScratchDirectory);
var sampler = new BatchSampler(corpus, contextLength: 64);

// Build a model and train it.
var model = new GptModel(
    vocabSize: tokeniser.VocabSize,
    embeddingDim: 128,
    numLayers: 4,
    numHeads: 4,
    maxSequenceLength: 64);

var optimizer = new AdamWOptimizer(model.Parameters(), learningRate: 3e-4f);
var trainer = new Trainer(model, sampler, optimizer);

trainer.Run(steps: 2000, batchSize: 8, onStep: (step, loss) =>
{
    if (step % 100 == 0)
    {
        Console.WriteLine($"step {step}: loss {loss:F4}");
    }
});

// Checkpoint the trained model.
ModelCheckpoint.Save(model, "model.checkpoint");

// Reload (possibly in a separate run/process, or via `dotnet run` in
// src/Chat for an interactive session) and generate text.
var loadedModel = ModelCheckpoint.Load("model.checkpoint");
var loadedTokeniser = BpeTokeniser.Load("vocab.bpe");

string generated = TextGenerator.Generate(
    loadedModel,
    loadedTokeniser,
    prompt: "Once upon a time",
    maxNewTokens: 100,
    options: new SamplingOptions { Temperature = 0.8f, TopK = 40 });

Console.WriteLine(generated);
```

Notes on the numbers above: `embeddingDim`/`numLayers`/`numHeads`/`steps`
are small, fast-to-run placeholders, not tuned hyperparameters — scale them
up (and expect training to take much longer, or reach for stage 8's
`--optimised`/CPU-parallelism if it does) for a model that produces
coherent text. `contextLength`/`maxSequenceLength` must match between the
sampler and the model.

### Running the example

```bash
dotnet new console -o MyTraining
cd MyTraining
dotnet add reference ../src/Tokeniser/Tokeniser.csproj ../src/Model/Model.csproj ../src/Training/Training.csproj ../src/Generation/Generation.csproj
# paste the code above into Program.cs, adjust CorpusPath/ScratchDirectory
dotnet run
```

## Project layout

- `src/Tokeniser/` — the BPE tokeniser (`BpeTokeniser.cs`), the
  pre-tokenisation chunker (`PreTokeniser.cs`), the bulk-encode result type
  (`EncodedCorpus.cs`), and the CLI (`TokeniserCli.cs`/`Program.cs`).
- `src/Common/MappedArray.cs` — disk-backed (memory-mapped) array type
  shared by the tokeniser and the tensor engine.
- `src/Tensor/` — the N-dimensional `Tensor` type, the `Variable`
  reverse-mode autodiff engine built on it, the optional `--optimised` CPU
  backend, and the optional `--gpu` ILGPU backend (`TensorBackend.cs`,
  `GpuContext.cs`, `Tensor.MatMul.Gpu.cs`).
- `src/Model/` — learned model components built on `Tensor`/`Variable`:
  token/positional embeddings, scaled dot-product and multi-head (causal)
  attention, layernorm, feed-forward, a GPT-2-style pre-norm
  `TransformerBlock`, `GptModel` (the full decoder-only model), and
  `GenerationCache` (the KV-cache).
- `src/Training/` — `TokenCorpus`/`BatchSampler` (pretraining's
  continuous-stream sliding windows), `CrossEntropyLoss` (plain and
  response-masked), `SgdOptimizer`/`AdamWOptimizer`, `ModelCheckpoint`,
  `Trainer` (the pretraining loop), and `SftExample`/`SftDataset`/`SftTrainer`
  (instruction tuning).
- `src/Pretrain/` — the pretraining CLI (`PretrainCli.cs`/`Program.cs`).
- `src/Sft/` — the instruction-tuning CLI (`SftCli.cs`/`Program.cs`).
- `src/Generation/` — `SamplingOptions`, `TokenSampler`, and
  `TextGenerator` (KV-cached autoregressive generation).
- `src/Chat/` — the interactive chat CLI.
- `examples/sft-example.jsonl` — a ready-to-use starter instruction-tuning dataset.
- `tests/*.Tests/` — one xUnit project per `src/` project above, mirroring
  its structure.

## Memory and disk footprint: a worked example

PLAN.md's standing "memory constraint" rule - disk-backed scratch structures
in place of large heap allocations, since this project has genuinely
OOM-killed itself before - isn't just a design aspiration. Here's what it
looks like in practice, measured on this machine (`/usr/bin/time -v` for
peak RAM, real byte counts for disk) against real corpora from 2 MB up to
100 MB, so the *trend* as input size grows is visible, not just one data
point:

**Stage 1 (tokeniser training)** - same target vocab size (1000) each time:

| Input text | Disk scratch (transient) | Peak RAM | `vocab.bpe` output |
|---|---|---|---|
| 2.0 MB | ~32 MB | ~194 MB | 3 KB |
| 4.0 MB | ~64 MB | ~230 MB | 3 KB |
| 10.0 MB | ~160 MB | ~373 MB | 3 KB |
| 100.0 MB | ~1600 MB | ~1.8 GB | 3 KB |

**Stage 6 (pretraining)** - same small 4-layer, 128-dim model architecture
each time, using the vocab/corpus from the matching row above (100 MB
skipped here - a full training run at that size takes long enough that the
tokeniser-training numbers above already make the point):

| Input text | Tokens | Disk (token corpus) | Peak RAM | `model.checkpoint` output |
|---|---|---|---|---|
| 2.0 MB | 502,147 | ~1.9 MB | ~279 MB | 3.3 MB |
| 4.0 MB | 1,004,276 | ~3.8 MB | ~239 MB | 3.3 MB |
| 10.0 MB | 2,510,680 | ~9.6 MB | ~324 MB | 3.3 MB |

The disk-scratch figures aren't estimates for this table specifically -
they're the exact numbers `TokeniserCli` itself prints before training
starts (`EstimateScratchBytes`: 16 bytes of scratch per byte of input, from
4 `int32` arrays sized to the corpus - see stage 1 above), and they scale
*exactly* linearly with input size, as the formula guarantees. The two
output artifacts (`vocab.bpe`, `model.checkpoint`) stay a *constant* size
across every row - they're sized by vocabulary/architecture, not by how
much text produced them.

**This table used to show peak RAM scaling linearly with input size, at
roughly 78 MB per MB of input text** (2 MB → ~216 MB, 4 MB → ~365 MB,
10 MB → ~840 MB) **- TASK-029 fixed the root cause, and these are the
re-measured numbers after that fix.** The bug was real: `BpeTokeniser.Train`
and `EncodeBulk` both went through `LinkedTokenStream.Build`, which read an
entire input file via `File.ReadAllText` into one heap-resident string
before anything reached disk-backed storage - unreclaimable managed-heap
memory, proportional to file size, sitting there for the whole build. The
fix (`PreTokeniser.Split(TextReader, bufferSize)`) reads and chunks the
file incrementally in bounded blocks instead, holding back only an actual
in-progress chunk (never more than one real word/run's worth of text)
across block boundaries - `Build` now runs two streaming passes instead of
one in-memory pass: a cheap first pass to count exact total bytes (sizing
the `MappedArray<T>`s precisely, no over-allocation), then a second pass to
fill them.

**Compared side by side, at 10 MB: ~840 MB peak RAM before, ~373 MB after**
for tokeniser training - the fix roughly halved it at this size, and the
gap widens with input size (the old code's growth was linear all the way
up; a 100 MB corpus under the old code would have extrapolated to roughly
18 GB, genuinely enough to trouble a typical desktop - the new code stays
under 2 GB at that same size). **Peak RAM still isn't perfectly flat**, and
that's expected, not a leftover version of the same bug: the
`Token`/`Next`/`Prev`/`PairNext` arrays are memory-mapped, but their pages
only become resident as `Build` actually writes into them, so the *active
working set* while filling a large corpus's worth of disk-backed arrays
still tracks corpus size too - just as `EstimateScratchBytes`'s existing
16-bytes-per-input-byte formula already predicts (100 MB × 16 bytes ≈
1.6 GB, close to the ~1.8 GB measured). The difference that actually
matters: those pages are *reclaimable* by the OS under memory pressure
(dropped if clean, written back if dirty) the same way any memory-mapped
file's pages are, unlike the old unreclaimable heap string - which is
exactly what the disk-backed design was supposed to buy in the first
place, and now genuinely does.

## Project status

Every stage above (1 through 11, including both optional `--optimised` and
`--gpu` backends) is built, tested, and runnable today — see
[TASK.md](TASK.md) for the task-by-task history and [PLAN.md](PLAN.md)'s
"Known limitations / deferred" section for the trade-offs (not bugs) that
were deliberately made along the way. Distributed (multi-machine) training
remains out of scope for now - not planned or tasked, though revisitable if
asked, the same way single-GPU execution (stage 11) just was; single-GPU
execution is no longer out of scope the way it once was.

The caveat that used to be repeated here — that the chat CLI doesn't
apply stage 9's prompt template or know when a response has "finished" —
is closed as of TASK-027/TASK-028: `--instruction-tuned` wraps each turn
in the exact template `SftDataset` uses and halts generation at the next
turn boundary instead of running on, and `--context-length` gives control
over how much conversation history is kept independent of the model's own
fixed `MaxSequenceLength`. What's still true, and isn't a bug: this
project's toy-sized demo models (tens of thousands of parameters, a few
hundred training steps, a few KB of corpus) are there to show each
mechanism working end to end, not to produce fluent or reliably on-topic
conversation — see stage 10's own transcripts for exactly what that looks
like in practice, warts included. TASK-029 fixed the memory/disk footprint
section's own headline finding: `File.ReadAllText` no longer holds an
entire corpus file on the unreclaimable managed heap during tokeniser
training/bulk-encoding (see the footprint section above for the
re-measured numbers and what, honestly, still isn't perfectly flat and
why). TASK-030 closed the SFT CLI scaling gap: `--epochs` (default 3)
sizes a training run from dataset size automatically - each epoch a
freshly shuffled full pass - instead of requiring `--batch-size` hand-tuned
to match the dataset the way the 6-example demo used to need; `--steps`
remains as a lower-level, unshuffled escape hatch, mutually exclusive with
`--epochs`.

TASK-031/032/033 added stage 11's optional GPU backend (ILGPU) - working,
tested, and (as of a follow-up fix) genuinely demonstrated on real GPU
hardware. The first version of this section carried a caveat that this
machine couldn't reach its own discrete AMD GPU at all - true at the time,
root-caused to one missing OS package (`ocl-icd-devel`, which provides the
unversioned `libOpenCL.so` symlink .NET's native-library probing needs;
the runtime library alone, already installed, wasn't enough). Installing
it fixed detection immediately, no code change - exactly what
`GpuContext`'s "prefer a real accelerator whenever one is reported" design
was supposed to make possible. Re-measured on the real AMD RX 6750 XT via
OpenCL: at this README's toy demo scale, GPU execution lands roughly even
with both CPU paths (see stage 11's table) - not a dramatic win, since
kernel-launch/transfer overhead still dominates actual compute at this
size, but a genuinely different and more complete finding than "GPU was
slower," which no longer holds now that a real GPU is reachable.

TASK-034/035/036 followed up on stage 11 once the real-hardware numbers
above landed roughly even instead of a clear GPU win: root-caused (not
disk storage - already excluded) to `MatMulGpu` allocating, transferring,
and freeing device buffers on every single matmul call. TASK-034 added
device-resident tensor storage; TASK-035 taught matmul to reuse an
already-resident operand instead of re-uploading it, explicitly *not*
making its output resident too (stated as a deliberate scope choice, not
a discovered gap - doing so would silently make every other op fall back
to a slow per-element device round trip); TASK-036 wired that mechanism
into an actual `--gpu-resident-weights` flag and measured the result
honestly: **dramatically slower (≈37×), not faster**, because backward's
`Transpose` and the optimizer's weight update aren't device-resident-aware
and pay that same per-element cost on every resident parameter, every
step. See stage 11 for the numbers and the full explanation - a real,
reportable finding about exactly where this optimization's limits are,
not a success story with the failure mode omitted.

No open gaps remain from this line of follow-up work - see TASK.md for the
full task-by-task history if scaling to a genuinely large corpus/dataset
(hundreds of MB, thousands of examples) raises something new.
