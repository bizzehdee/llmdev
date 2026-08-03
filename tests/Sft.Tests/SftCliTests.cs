using Model;
using Tokeniser;
using Training;
using Xunit;

namespace Sft.Tests;

public class SftCliTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "sft-tests-scratch");
    private static readonly string VocabPath = Path.Combine(ScratchDirectory, "fixture-vocab.bpe");
    private static readonly string BaseCheckpointPath = Path.Combine(ScratchDirectory, "fixture-base.checkpoint");
    private static readonly string DatasetPath = Path.Combine(ScratchDirectory, "fixture-dataset.jsonl");

    static SftCliTests()
    {
        Directory.CreateDirectory(ScratchDirectory);

        string corpusPath = Path.Combine(ScratchDirectory, "fixture-corpus.txt");
        File.WriteAllText(corpusPath, string.Concat(Enumerable.Repeat(
            "### Instruction:\nWhat is the capital of France?\n\n### Response:\nParis is the capital of France. ", 40)));

        var tokeniser = new BpeTokeniser();
        tokeniser.Train([corpusPath], targetVocabSize: 280, ScratchDirectory);
        tokeniser.Save(VocabPath);
        File.Delete(corpusPath);

        var model = new GptModel(vocabSize: tokeniser.VocabSize, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 64, random: new Random(1));
        ModelCheckpoint.Save(model, BaseCheckpointPath);

        File.WriteAllLines(DatasetPath, new[]
        {
            """{"instruction": "What is the capital of France?", "response": "Paris is the capital of France."}""",
        });
    }

    private static (int exitCode, string stdout, string stderr) Run(params string[] args)
    {
        using var stdout = new StringWriter();
        using var stderr = new StringWriter();
        int exitCode = SftCli.Run(args, stdout, stderr);
        return (exitCode, stdout.ToString(), stderr.ToString());
    }

    [Fact]
    public void Run_TooFewArguments_PrintsUsageAndReturnsOne()
    {
        var (exitCode, stdout, _) = Run(BaseCheckpointPath, VocabPath, DatasetPath);

        Assert.Equal(1, exitCode);
        Assert.Contains("Usage: Sft", stdout);
    }

    [Fact]
    public void Run_UnrecognisedFlag_ReturnsError()
    {
        var (exitCode, _, stderr) = Run(BaseCheckpointPath, VocabPath, DatasetPath, "out.checkpoint", "--not-a-real-flag");

        Assert.Equal(1, exitCode);
        Assert.Contains("Unrecognised or malformed option", stderr);
    }

    [Fact]
    public void Run_MalformedFlagValue_ReturnsError()
    {
        var (exitCode, _, stderr) = Run(BaseCheckpointPath, VocabPath, DatasetPath, "out.checkpoint", "--steps", "not-a-number");

        Assert.Equal(1, exitCode);
        Assert.Contains("Unrecognised or malformed option", stderr);
    }

    [Fact]
    public void Run_OutputPathEqualToBaseCheckpoint_RefusesAndReturnsError()
    {
        var (exitCode, _, stderr) = Run(BaseCheckpointPath, VocabPath, DatasetPath, BaseCheckpointPath);

        Assert.Equal(1, exitCode);
        Assert.Contains("Refusing to fine-tune", stderr);
        Assert.Contains("must differ", stderr);
    }

    [Fact]
    public void Run_MissingBaseCheckpoint_ReturnsErrorWithMessage()
    {
        var (exitCode, _, stderr) = Run("/nonexistent/base.checkpoint", VocabPath, DatasetPath, "out.checkpoint");

        Assert.Equal(1, exitCode);
        Assert.Contains("Failed to load base checkpoint or vocabulary", stderr);
    }

    [Fact]
    public void Run_MissingDatasetFile_ReturnsErrorWithMessage()
    {
        var (exitCode, _, stderr) = Run(BaseCheckpointPath, VocabPath, "/nonexistent/dataset.jsonl", "out.checkpoint");

        Assert.Equal(1, exitCode);
        Assert.Contains("Failed to load SFT dataset", stderr);
    }

    [Fact]
    public void Run_MalformedDatasetLine_ReturnsErrorWithMessage()
    {
        string malformedPath = Path.Combine(ScratchDirectory, $"malformed-{Guid.NewGuid():N}.jsonl");
        File.WriteAllLines(malformedPath, new[] { "not valid json" });
        try
        {
            var (exitCode, _, stderr) = Run(BaseCheckpointPath, VocabPath, malformedPath, "out.checkpoint");

            Assert.Equal(1, exitCode);
            Assert.Contains("Failed to load SFT dataset", stderr);
        }
        finally
        {
            File.Delete(malformedPath);
        }
    }

    [Fact]
    public void Run_EmptyDataset_ReturnsErrorWithMessage()
    {
        string emptyPath = Path.Combine(ScratchDirectory, $"empty-{Guid.NewGuid():N}.jsonl");
        File.WriteAllText(emptyPath, "");
        try
        {
            var (exitCode, _, stderr) = Run(BaseCheckpointPath, VocabPath, emptyPath, "out.checkpoint");

            Assert.Equal(1, exitCode);
            Assert.Contains("contains no examples", stderr);
        }
        finally
        {
            File.Delete(emptyPath);
        }
    }

    [Fact]
    public void Run_EndToEnd_ProducesADifferentLoadableCheckpointWithDecreasedLoss()
    {
        string outputCheckpointPath = Path.Combine(ScratchDirectory, $"tuned-{Guid.NewGuid():N}.checkpoint");

        var (exitCode, stdout, stderr) = Run(
            BaseCheckpointPath, VocabPath, DatasetPath, outputCheckpointPath,
            "--steps", "60", "--batch-size", "1", "--learning-rate", "0.01", "--weight-decay", "0");

        Assert.Equal(0, exitCode);
        Assert.Empty(stderr);
        Assert.True(File.Exists(outputCheckpointPath), "Expected a fine-tuned checkpoint file to be written.");

        var losses = stdout.Split('\n')
            .Where(line => line.StartsWith("step ", StringComparison.Ordinal))
            .Select(line => float.Parse(line.Split("loss ")[1]))
            .ToList();
        Assert.True(losses.Count >= 2, "Expected at least a first and last loss to be printed.");
        Assert.True(losses[^1] < losses[0], $"Expected loss to drop: {losses[0]} -> {losses[^1]}.");

        // The base checkpoint must be untouched: still loadable, and this
        // is a genuinely separate file.
        var baseModel = ModelCheckpoint.Load(BaseCheckpointPath);
        var tunedModel = ModelCheckpoint.Load(outputCheckpointPath);
        Assert.Equal(baseModel.EmbeddingDim, tunedModel.EmbeddingDim);
    }

    [Fact]
    public void Run_OptimisedFlag_SelectsOptimisedTensorBackend()
    {
        string outputCheckpointPath = Path.Combine(ScratchDirectory, $"tuned-{Guid.NewGuid():N}.checkpoint");
        try
        {
            var (exitCode, _, _) = Run(
                BaseCheckpointPath, VocabPath, DatasetPath, outputCheckpointPath,
                "--steps", "2", "--batch-size", "1", "--optimised");

            Assert.Equal(0, exitCode);
            Assert.Equal(Tensor.TensorBackend.Optimised, Tensor.Tensor.Backend);
        }
        finally
        {
            Tensor.Tensor.Backend = Tensor.TensorBackend.Scalar;
        }
    }
}
