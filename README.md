# llmdev

A personal project to learn how LLMs work by building one from first
principles in C#/.NET — no ML, tokenisation, or tensor/autodiff libraries
(one narrow, explicitly-scoped exception: an opt-in `--optimised` fast path
for tensor math, stage 9 below — off by default, and not a replacement for
the hand-written implementation it speeds up). Every mechanism (tokeniser,
tensor math, autodiff, attention, training loop, generation, instruction
tuning) is written by hand so it's understood, not a black box behind a
library call.

This README is a lesson plan: one section per stage, in the order they were
built, each covering what problem the stage solves, what's actually
happening conceptually, which source files to go read, and what to run to
see it working. Every stage below is built and runnable today — see
[Project status](#project-status) for the one honest caveat worth knowing
about before you expect too much of it as a chatbot.

See [PLAN.md](PLAN.md) for the full roadmap/rationale and [TASK.md](TASK.md)
for the task-by-task history of how it got built.

## Requirements

- [.NET 10 SDK](https://dotnet.microsoft.com/download)

## Stage 1 — Tokeniser

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
dotnet run -- 500 ../../sample.txt          # train on a single file
dotnet run -- 2000 ~/Documents/some-corpus  # or every .txt file in a directory
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

**Source files:** `src/Tensor/Tensor.*.cs` (the tensor type, split across
files by operation group), `src/Tensor/Variable*.cs` (the autodiff layer),
`src/Tensor/{HeapFloatBuffer,MappedFloatBuffer,IFloatBuffer}.cs` (the
storage abstraction).

**Run it:** there's no CLI for this stage — it's a library other stages
build on. The tests *are* the way to see it working, including
finite-difference gradient checks (numerically perturb an input, compare
against the analytic gradient `Backward()` computed) rather than only
hand-derived expected values:

```bash
cd tests/Tensor.Tests && dotnet test
```

## Stage 3 — Embeddings

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

**Run it:** again a library piece, exercised by:

```bash
cd tests/Model.Tests && dotnet test
```

(covers embeddings, along with every other stage-4/5 component below — one
test project for the whole `Model` project.)

## Stage 4 — Attention + transformer block

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

**Run it:** `cd tests/Model.Tests && dotnet test` (as above) — includes an
end-to-end check that causal masking actually holds through a full block
(changing a later token's value must never change an earlier position's
output).

## Stage 5 — Model assembly

**Problem it solves:** stages 2–4 are all *pieces*; something has to stack
them into an actual language model.

**What's happening:** `GptModel` is a decoder-only, GPT-2-style model: token
+ positional embeddings, a stack of causal transformer blocks, a final
layernorm, and an output projection back to logits over the vocabulary. The
output projection reuses (rather than duplicates) the token embedding's
weight matrix, transposed — "weight tying", which halves what would
otherwise be two separate `[vocabSize, embeddingDim]`-sized matrices and is
a reasonable prior besides (a token's input representation and its output
"how likely is this next" score plausibly share structure).

**Source files:** `src/Model/GptModel.cs`.

**Run it:** `cd tests/Model.Tests && dotnet test` (as above) — includes
finite-difference gradient checks through the *entire* model, not just
individual layers.

## Stage 6 — Training loop

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
`src/Training/ModelCheckpoint.cs`, `src/Training/Trainer.cs`.

**Run it:** see the [full worked example](#putting-it-together-training-and-generating-from-code)
below — training isn't exposed as a CLI (there's no single obviously-right
set of hyperparameters/corpus to hardcode into one), so it's driven from a
short C# snippet instead.

```bash
cd tests/Training.Tests && dotnet test
```

includes a genuine end-to-end proof: loss measurably drops over training
steps on a small, deliberately repetitive corpus.

## Stage 7 — Generation

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

**Run it:** see the [full worked example](#putting-it-together-training-and-generating-from-code)
below, or stage 8's interactive CLI for a more hands-on way to try it.

```bash
cd tests/Generation.Tests && dotnet test
```

verifies sampling correctness (deterministic greedy, statistical
distribution checks for temperature/top-k/top-p) and, critically, that the
KV-cached generation path produces *exactly* the same output as a
from-scratch, non-cached recompute at every step — not just "plausibly
similar text."

## Stage 8 — Interactive chat CLI

**Problem it solves:** stages 1–7 are only usable via a one-off C# snippet
(the [worked example](#putting-it-together-training-and-generating-from-code)
below) — there's no way to just sit down and talk to a trained model
turn-by-turn.

**What's happening:** loads a saved `ModelCheckpoint` + tokeniser vocab,
then loops: read a line of input, generate a continuation, print it,
repeat. Conversation state is a single growing token-id sequence (not
re-encoded text each turn), so multi-turn context accumulates correctly.

**Honest expectation-setting, not a bug:** this is a *raw
next-token-prediction model*, not an instruction-tuned assistant, unless
you've separately fine-tuned it per stage 10 below — and even then, the CLI
itself doesn't automatically wrap what you type in stage 10's prompt
template, so a stage-10-tuned model won't behave as intended unless you
type your input already shaped the way that template expects. Without
fine-tuning, it will continue text in the style of whatever it was trained
on, not necessarily answer questions or follow instructions. The CLI says
this up front, not just here.

**Source files:** `src/Chat/ChatCli.cs`.

**Run it:**

```bash
cd src/Chat
dotnet run -- <checkpoint-path> <vocab-path> [--temperature <f>] [--top-k <n>] [--top-p <f>] [--max-new-tokens <n>] [--optimised]
```

```bash
cd tests/Chat.Tests && dotnet test
```

## Stage 9 — Optional optimised math backend (`--optimised`)

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

**Run it:** pass `--optimised` to the [chat CLI](#stage-8--interactive-chat-cli)
above, or set `Tensor.Backend = TensorBackend.Optimised` in your own code
before calling into the model. Verified by running the *same* test suite
(including every stage-2 finite-difference gradient check) against both
backends and confirming equivalent results — see the `[Theory]`-parametrised
tests in `tests/Tensor.Tests/{TensorTests,VariableTests}.cs`.

## Stage 10 — Instruction tuning (SFT)

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

**Source files:** `src/Training/{SftExample,SftDataset,SftTrainer}.cs`,
`src/Training/CrossEntropyLoss.cs` (`ComputeMasked`).

**Run it:** there's no CLI for this stage either, for the same reason as
stage 6 — it's driven from a short C# snippet analogous to the
[worked example](#putting-it-together-training-and-generating-from-code)
below, swapping `Trainer`/`TokenCorpus`/`BatchSampler` for
`SftTrainer`/`SftDataset`.

```bash
cd tests/Training.Tests && dotnet test
```

includes an end-to-end proof that loss actually drops on a small repetitive
instruction/response pattern, mirroring stage 6's pretraining equivalent.

## Putting it together: training and generating from code

There's a CLI for the tokeniser (stage 1) and the chat CLI (stage 8), but
pretraining and instruction tuning (stages 6, 10) are library-only — no
single hardcoded set of hyperparameters/corpus/dataset would be right for
everyone. The example below walks through stages 2–7 (pretraining +
generation); stage 10's fine-tuning loop looks the same shape, just with
`SftDataset.Load` + `SftTrainer` in place of `TokenCorpus`/`BatchSampler` +
`Trainer`.

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
up (and expect training to take much longer, or reach for stage 9's
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
  reverse-mode autodiff engine built on it, and the optional `--optimised`
  backend (`TensorBackend.cs`).
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
- `src/Generation/` — `SamplingOptions`, `TokenSampler`, and
  `TextGenerator` (KV-cached autoregressive generation).
- `src/Chat/` — the interactive chat CLI.
- `tests/*.Tests/` — one xUnit project per `src/` project above, mirroring
  its structure.

## Project status

Every stage above (1 through 10, plus the optional `--optimised` backend)
is built, tested, and runnable today — see [TASK.md](TASK.md) for the
task-by-task history and [PLAN.md](PLAN.md)'s "Known limitations /
deferred" section for the trade-offs (not bugs) that were deliberately made
along the way. The one thing genuinely out of scope, not just "not done
yet": GPU/distributed training — this project targets a single
CPU-only machine throughout.

The one honest caveat worth repeating from stage 8: fine-tuning a model
(stage 10) makes it more likely to behave like a useful conversational
assistant, but the chat CLI doesn't automatically format what you type
using stage 10's prompt template — for a fine-tuned model to behave as
intended, whatever prompts it needs to be shaped the way that template
expects, not just typed as a bare instruction.
