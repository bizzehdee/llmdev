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
        tokens.Add(TokenSampler.Sample(ExtractRow(logits, logits.Shape[0] - 1), options, random));

        for (int step = 1; step < maxNewTokens; step++)
        {
            if (tokens.Count > model.MaxSequenceLength)
            {
                cache.Reset();
                context = tokens.Skip(tokens.Count - model.MaxSequenceLength).ToArray();
                logits = model.ForwardIncremental(context, cache).Value;
            }
            else
            {
                logits = model.ForwardIncremental([tokens[^1]], cache).Value;
            }

            tokens.Add(TokenSampler.Sample(ExtractRow(logits, logits.Shape[0] - 1), options, random));
        }

        return tokens;
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
