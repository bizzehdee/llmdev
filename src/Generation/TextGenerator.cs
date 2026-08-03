using Model;
using Tokeniser;
using TensorValue = Tensor.Tensor;

namespace Generation;

/// <summary>
/// Runs a trained (or in-progress) <see cref="GptModel"/> autoregressively:
/// repeatedly predict the next token, sample one (per
/// <see cref="SamplingOptions"/>), append it, repeat. Uses a KV-cache
/// (TASK-020, <see cref="GenerationCache"/>) so each step after the
/// initial prompt only computes the one new token's Q/K/V instead of
/// recomputing every layer over the whole growing context from scratch -
/// mathematically identical to the old full-recompute approach at every
/// step (see the KV-cache correctness tests), just without the repeated
/// work.
/// </summary>
public static class TextGenerator
{
    /// <summary>
    /// Extends <paramref name="promptTokenIds"/> by up to
    /// <paramref name="maxNewTokens"/> tokens. Once the sequence would
    /// exceed the model's context window, only the most recent
    /// <c>MaxSequenceLength</c> tokens are fed to the model (a sliding
    /// window) - generation keeps going, just with truncated context. A
    /// KV-cache can't simply be shifted when that happens (its positions
    /// are tied to absolute offsets), so a sliding-window step rebuilds
    /// the cache from scratch for the truncated window instead - the same
    /// one-step cost the old always-recompute approach paid on *every*
    /// step, just now confined to the (rare) truncation steps.
    /// </summary>
    public static List<int> GenerateTokenIds(GptModel model, int[] promptTokenIds, int maxNewTokens, SamplingOptions options, Random? random = null)
    {
        random ??= new Random();
        var tokens = new List<int>(promptTokenIds);
        if (maxNewTokens <= 0)
        {
            return tokens;
        }

        using var cache = new GenerationCache(model.NumLayers);
        var context = tokens.Count > model.MaxSequenceLength
            ? tokens.Skip(tokens.Count - model.MaxSequenceLength).ToArray()
            : tokens.ToArray();
        var logits = model.ForwardIncremental(context, cache).Value;

        for (int step = 0; step < maxNewTokens; step++)
        {
            if (step > 0)
            {
                (logits, context) = Advance(model, cache, tokens, context);
            }
            tokens.Add(TokenSampler.Sample(ExtractRow(logits, logits.Shape[0] - 1), options, random));
        }

        return tokens;
    }

    /// <summary>
    /// Like <see cref="GenerateTokenIds(GptModel, int[], int, SamplingOptions, Random?)"/>,
    /// but halts before <paramref name="maxNewTokens"/> is reached once the
    /// decoded text of the *newly generated* tokens contains
    /// <paramref name="stopSequence"/>, trimming the return value back to
    /// exclude the stop sequence and anything after it. Needed for
    /// TASK-027's instruction-tuned chat mode: a response shouldn't run on
    /// into a hallucinated next "### Instruction:" turn. This can't be a
    /// token-id check - a BPE tokeniser's merges mean there's no fixed set
    /// of token ids that reliably spells a given piece of text, only
    /// decoded text does - so trimming re-encodes the truncated text
    /// rather than slicing the raw generated token ids.
    /// </summary>
    public static List<int> GenerateTokenIdsUntilStopSequence(GptModel model, BpeTokeniser tokeniser, int[] promptTokenIds, int maxNewTokens, string stopSequence, SamplingOptions options, Random? random = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(stopSequence);

        random ??= new Random();
        var tokens = new List<int>(promptTokenIds);
        if (maxNewTokens <= 0)
        {
            return tokens;
        }

        using var cache = new GenerationCache(model.NumLayers);
        var context = tokens.Count > model.MaxSequenceLength
            ? tokens.Skip(tokens.Count - model.MaxSequenceLength).ToArray()
            : tokens.ToArray();
        var logits = model.ForwardIncremental(context, cache).Value;

        int promptLength = tokens.Count;
        var generated = new List<int>();

        for (int step = 0; step < maxNewTokens; step++)
        {
            if (step > 0)
            {
                (logits, context) = Advance(model, cache, tokens, context);
            }

            int next = TokenSampler.Sample(ExtractRow(logits, logits.Shape[0] - 1), options, random);
            tokens.Add(next);
            generated.Add(next);

            string decoded = tokeniser.Decode(generated);
            int stopIndex = decoded.IndexOf(stopSequence, StringComparison.Ordinal);
            if (stopIndex >= 0)
            {
                var trimmedTokenIds = tokeniser.Encode(decoded[..stopIndex]);
                var result = tokens.Take(promptLength).ToList();
                result.AddRange(trimmedTokenIds);
                return result;
            }
        }

        return tokens;
    }

    /// <summary>One incremental decode step: rebuilds the cache from a sliding window if the context overflowed, otherwise feeds just the last token.</summary>
    private static (TensorValue logits, int[] context) Advance(GptModel model, GenerationCache cache, List<int> tokens, int[] context)
    {
        if (tokens.Count > model.MaxSequenceLength)
        {
            cache.Reset();
            context = tokens.Skip(tokens.Count - model.MaxSequenceLength).ToArray();
            return (model.ForwardIncremental(context, cache).Value, context);
        }

        return (model.ForwardIncremental([tokens[^1]], cache).Value, context);
    }

    /// <summary>Encodes <paramref name="prompt"/>, generates, and decodes the full (prompt + generated) sequence back to text.</summary>
    public static string Generate(GptModel model, BpeTokeniser tokeniser, string prompt, int maxNewTokens, SamplingOptions options, Random? random = null)
    {
        var promptTokenIds = tokeniser.Encode(prompt).ToArray();
        var tokens = GenerateTokenIds(model, promptTokenIds, maxNewTokens, options, random);
        return tokeniser.Decode(tokens);
    }

    private static float[] ExtractRow(TensorValue logits, int row)
    {
        int vocabSize = logits.Shape[1];
        var result = new float[vocabSize];
        for (int v = 0; v < vocabSize; v++)
        {
            result[v] = logits[row, v];
        }
        return result;
    }
}
