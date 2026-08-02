using Model;
using Tokeniser;
using TensorValue = Tensor.Tensor;

namespace Generation;

/// <summary>
/// Runs a trained (or in-progress) <see cref="GptModel"/> autoregressively:
/// repeatedly predict the next token, sample one (per
/// <see cref="SamplingOptions"/>), append it, repeat. No KV-cache - each
/// step recomputes a full forward pass over the whole context, which is
/// the simple, correct thing rather than the fast thing (this is a
/// learning project's generation loop, not a production inference
/// server); noted in TASK.md as a known, deliberate limitation.
/// </summary>
public static class TextGenerator
{
    /// <summary>
    /// Extends <paramref name="promptTokenIds"/> by up to
    /// <paramref name="maxNewTokens"/> tokens. Once the sequence would
    /// exceed the model's context window, only the most recent
    /// <c>MaxSequenceLength</c> tokens are fed to the model (a sliding
    /// window) - generation keeps going, just with truncated context.
    /// </summary>
    public static List<int> GenerateTokenIds(GptModel model, int[] promptTokenIds, int maxNewTokens, SamplingOptions options, Random? random = null)
    {
        random ??= new Random();
        var tokens = new List<int>(promptTokenIds);

        for (int step = 0; step < maxNewTokens; step++)
        {
            var context = tokens.Count > model.MaxSequenceLength
                ? tokens.Skip(tokens.Count - model.MaxSequenceLength).ToArray()
                : tokens.ToArray();

            var logits = model.Forward(context).Value;
            var nextTokenLogits = ExtractRow(logits, context.Length - 1);

            tokens.Add(TokenSampler.Sample(nextTokenLogits, options, random));
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
