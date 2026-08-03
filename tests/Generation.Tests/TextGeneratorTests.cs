using Model;
using Tokeniser;
using Xunit;

namespace Generation.Tests;

public class TextGeneratorTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "generation-tests-scratch");

    static TextGeneratorTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    [Fact]
    public void GenerateTokenIds_AppendsExactlyMaxNewTokens()
    {
        var model = new GptModel(vocabSize: 10, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 20, random: new Random(1));
        int[] prompt = [1, 2, 3];

        var result = TextGenerator.GenerateTokenIds(model, prompt, maxNewTokens: 5, SamplingOptions.Greedy());

        Assert.Equal(8, result.Count);
        Assert.Equal(prompt, result.Take(3));
    }

    [Fact]
    public void GenerateTokenIds_Greedy_IsDeterministic()
    {
        var model = new GptModel(vocabSize: 10, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 20, random: new Random(1));
        int[] prompt = [1, 2, 3];

        var first = TextGenerator.GenerateTokenIds(model, prompt, maxNewTokens: 10, SamplingOptions.Greedy());
        var second = TextGenerator.GenerateTokenIds(model, prompt, maxNewTokens: 10, SamplingOptions.Greedy());

        Assert.Equal(first, second);
    }

    [Fact]
    public void GenerateTokenIds_AllTokensAreWithinVocabRange()
    {
        var model = new GptModel(vocabSize: 12, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 20, random: new Random(1));
        int[] prompt = [0];
        var options = new SamplingOptions { Temperature = 1.2f, TopK = 5 };

        var result = TextGenerator.GenerateTokenIds(model, prompt, maxNewTokens: 15, options, new Random(2));

        Assert.All(result, id => Assert.InRange(id, 0, 11));
    }

    [Fact]
    public void GenerateTokenIds_ExceedingMaxSequenceLengthKeepsGeneratingViaSlidingWindow()
    {
        var model = new GptModel(vocabSize: 10, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(1));
        int[] prompt = [1, 2];

        var result = TextGenerator.GenerateTokenIds(model, prompt, maxNewTokens: 10, SamplingOptions.Greedy());

        Assert.Equal(12, result.Count);
        Assert.All(result, id => Assert.InRange(id, 0, 9));
    }

    [Fact]
    public void GenerateTokenIds_MatchesANonCachedFullRecomputeReferenceImplementation()
    {
        // TASK-020's own bar applied at the public API level: the
        // KV-cached generation loop must produce identical output to the
        // old approach of recomputing the full forward pass from scratch
        // every step - reimplemented here directly (not by calling
        // TextGenerator itself) as the reference to compare against.
        var model = new GptModel(vocabSize: 12, embeddingDim: 8, numLayers: 2, numHeads: 2, maxSequenceLength: 6, random: new Random(3));
        int[] prompt = [2, 5, 1];
        var options = SamplingOptions.Greedy();

        var cachedResult = TextGenerator.GenerateTokenIds(model, prompt, maxNewTokens: 6, options);
        var referenceResult = GenerateViaFullRecompute(model, prompt, maxNewTokens: 6, options);

        Assert.Equal(referenceResult, cachedResult);
    }

    private static List<int> GenerateViaFullRecompute(GptModel model, int[] promptTokenIds, int maxNewTokens, SamplingOptions options)
    {
        var tokens = new List<int>(promptTokenIds);
        var random = new Random();

        for (int step = 0; step < maxNewTokens; step++)
        {
            var context = tokens.Count > model.MaxSequenceLength
                ? tokens.Skip(tokens.Count - model.MaxSequenceLength).ToArray()
                : tokens.ToArray();

            var logits = model.Forward(context).Value;
            int vocabSize = logits.Shape[1];
            var row = new float[vocabSize];
            for (int v = 0; v < vocabSize; v++)
            {
                row[v] = logits[context.Length - 1, v];
            }

            tokens.Add(TokenSampler.Sample(row, options, random));
        }

        return tokens;
    }

    [Fact]
    public void Generate_ReturnsTextStartingWithTheDecodedPrompt()
    {
        // Decode is plain concatenation of each token's bytes in order (see
        // BpeTokeniser.Decode), so the full generated string must start with
        // exactly the decoded prompt, regardless of what the model produces.
        var trainPath = Path.Combine(ScratchDirectory, $"corpus-{Guid.NewGuid():N}.txt");
        File.WriteAllText(trainPath, "the quick brown fox jumps over the lazy dog. the dog barks at the fox.");
        var tokeniser = new BpeTokeniser();
        tokeniser.Train([trainPath], targetVocabSize: 280, ScratchDirectory);
        File.Delete(trainPath);

        var model = new GptModel(vocabSize: tokeniser.VocabSize, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 30, random: new Random(1));

        string prompt = "the fox";
        string generated = TextGenerator.Generate(model, tokeniser, prompt, maxNewTokens: 10, SamplingOptions.Greedy());

        string decodedPrompt = tokeniser.Decode(tokeniser.Encode(prompt));
        Assert.StartsWith(decodedPrompt, generated);
    }
}
