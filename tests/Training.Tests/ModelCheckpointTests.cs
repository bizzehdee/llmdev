using Model;
using Xunit;

namespace Training.Tests;

public class ModelCheckpointTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "training-tests-scratch");

    static ModelCheckpointTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    [Fact]
    public void SaveThenLoad_ProducesIdenticalParameterValues()
    {
        var model = new GptModel(vocabSize: 12, embeddingDim: 8, numLayers: 2, numHeads: 2, maxSequenceLength: 10, random: new Random(1));
        string path = Path.Combine(ScratchDirectory, $"checkpoint-{Guid.NewGuid():N}.bin");

        try
        {
            ModelCheckpoint.Save(model, path);
            var loaded = ModelCheckpoint.Load(path);

            var originalParams = model.Parameters();
            var loadedParams = loaded.Parameters();
            Assert.Equal(originalParams.Count, loadedParams.Count);
            for (int i = 0; i < originalParams.Count; i++)
            {
                Assert.Equal(originalParams[i].Value.Shape, loadedParams[i].Value.Shape);
                Assert.Equal(originalParams[i].Value.ToArray(), loadedParams[i].Value.ToArray());
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void SaveThenLoad_ProducesIdenticalForwardPassOutput()
    {
        var model = new GptModel(vocabSize: 12, embeddingDim: 8, numLayers: 2, numHeads: 2, maxSequenceLength: 10, random: new Random(1));
        int[] tokenIds = [1, 5, 3, 2];
        string path = Path.Combine(ScratchDirectory, $"checkpoint-{Guid.NewGuid():N}.bin");

        try
        {
            var logitsBefore = model.Forward(tokenIds).Value.ToArray();

            ModelCheckpoint.Save(model, path);
            var loaded = ModelCheckpoint.Load(path);
            var logitsAfter = loaded.Forward(tokenIds).Value.ToArray();

            Assert.Equal(logitsBefore, logitsAfter);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_RestoresArchitectureHyperparameters()
    {
        var model = new GptModel(vocabSize: 12, embeddingDim: 8, numLayers: 3, numHeads: 4, maxSequenceLength: 10, feedForwardHiddenDim: 20, random: new Random(1));
        string path = Path.Combine(ScratchDirectory, $"checkpoint-{Guid.NewGuid():N}.bin");

        try
        {
            ModelCheckpoint.Save(model, path);
            var loaded = ModelCheckpoint.Load(path);

            Assert.Equal(12, loaded.VocabSize);
            Assert.Equal(8, loaded.EmbeddingDim);
            Assert.Equal(3, loaded.NumLayers);
            Assert.Equal(4, loaded.NumHeads);
            Assert.Equal(10, loaded.MaxSequenceLength);
            Assert.Equal(20, loaded.Blocks[0].FeedForward.HiddenDim);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_ChangedParameterCountInFileThrows()
    {
        var model = new GptModel(vocabSize: 6, embeddingDim: 4, numLayers: 1, numHeads: 2, maxSequenceLength: 5, random: new Random(1));
        string path = Path.Combine(ScratchDirectory, $"checkpoint-{Guid.NewGuid():N}.bin");

        try
        {
            ModelCheckpoint.Save(model, path);

            // Corrupt the saved parameter count (the int right after the 6 header ints).
            var bytes = File.ReadAllBytes(path);
            const int headerInts = 6;
            int countOffset = headerInts * sizeof(int);
            BitConverter.GetBytes(999).CopyTo(bytes, countOffset);
            File.WriteAllBytes(path, bytes);

            Assert.Throws<InvalidOperationException>(() => ModelCheckpoint.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
