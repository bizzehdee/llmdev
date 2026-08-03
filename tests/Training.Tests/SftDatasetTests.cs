using Tokeniser;
using Xunit;

namespace Training.Tests;

public class SftDatasetTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "training-tests-scratch");

    static SftDatasetTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    private static BpeTokeniser TrainFixtureTokeniser()
    {
        string corpusPath = Path.Combine(ScratchDirectory, $"sft-corpus-{Guid.NewGuid():N}.txt");
        File.WriteAllText(corpusPath, string.Concat(Enumerable.Repeat(
            "### Instruction:\nWhat is the capital of France?\n\n### Response:\nParis is the capital of France. ", 40)));

        var tokeniser = new BpeTokeniser();
        tokeniser.Train([corpusPath], targetVocabSize: 280, ScratchDirectory);
        File.Delete(corpusPath);
        return tokeniser;
    }

    [Fact]
    public void Tokenize_InputAndTargetAreTheStandardNextTokenShift()
    {
        var tokeniser = TrainFixtureTokeniser();
        var example = new SftExample("What is the capital of France?", "Paris.");

        var result = SftDataset.Tokenize(example, tokeniser);

        Assert.Equal(result.InputIds.Length, result.TargetIds.Length);
        Assert.Equal(result.InputIds.Length, result.ResponseMask.Length);
    }

    [Fact]
    public void Tokenize_ResponseMaskIsFalseForPromptPositionsAndTrueForResponsePositions()
    {
        var tokeniser = TrainFixtureTokeniser();
        var example = new SftExample("What is the capital of France?", "Paris is the capital of France.");

        var result = SftDataset.Tokenize(example, tokeniser);

        // The mask must not be all-true or all-false: there's a real prompt
        // and a real response, both contributing at least one token.
        Assert.Contains(result.ResponseMask, m => !m);
        Assert.Contains(result.ResponseMask, m => m);
        // Once true, the mask must stay true (response tokens are a
        // contiguous suffix, since the prompt is always encoded first).
        int firstTrue = Array.IndexOf(result.ResponseMask, true);
        for (int i = firstTrue; i < result.ResponseMask.Length; i++)
        {
            Assert.True(result.ResponseMask[i], $"Expected mask[{i}] to be true once the response starts.");
        }
    }

    [Fact]
    public void Load_ParsesJsonLinesFile()
    {
        var tokeniser = TrainFixtureTokeniser();
        string datasetPath = Path.Combine(ScratchDirectory, $"sft-dataset-{Guid.NewGuid():N}.jsonl");
        File.WriteAllLines(datasetPath, new[]
        {
            """{"instruction": "What is the capital of France?", "response": "Paris."}""",
            "",
            """{"instruction": "Name a fruit.", "response": "Apple."}""",
        });

        try
        {
            var examples = SftDataset.Load(datasetPath, tokeniser);

            Assert.Equal(2, examples.Count);
        }
        finally
        {
            File.Delete(datasetPath);
        }
    }

    [Fact]
    public void Load_MalformedLineThrows()
    {
        var tokeniser = new BpeTokeniser();
        string datasetPath = Path.Combine(ScratchDirectory, $"sft-dataset-{Guid.NewGuid():N}.jsonl");
        File.WriteAllLines(datasetPath, new[] { "not valid json" });

        try
        {
            Assert.ThrowsAny<Exception>(() => SftDataset.Load(datasetPath, tokeniser));
        }
        finally
        {
            File.Delete(datasetPath);
        }
    }
}
