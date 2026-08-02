using Model;
using Xunit;

namespace Training.Tests;

public class TrainerTests
{
    private static readonly string ScratchDirectory = Path.Combine(Path.GetTempPath(), "training-tests-scratch");

    static TrainerTests()
    {
        Directory.CreateDirectory(ScratchDirectory);
    }

    private static (GptModel model, BatchSampler sampler) BuildTinySetup(Random random)
    {
        // A short, repetitive corpus (easy for a tiny model to actually
        // learn something about within a handful of steps) rather than
        // random noise, which a model this small couldn't meaningfully fit.
        var tokens = new List<int>();
        for (int i = 0; i < 40; i++)
        {
            tokens.AddRange([1, 2, 3, 4]);
        }
        var corpus = new TokenCorpus(tokens, ScratchDirectory);
        var sampler = new BatchSampler(corpus, contextLength: 4, random);
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: random);
        return (model, sampler);
    }

    [Fact]
    public void Step_ReturnsAFiniteNonNegativeLoss()
    {
        var (model, sampler) = BuildTinySetup(new Random(1));
        var optimizer = new SgdOptimizer(model.Parameters(), learningRate: 0.01f);
        var trainer = new Trainer(model, sampler, optimizer);

        float loss = trainer.Step(batchSize: 4);

        Assert.True(loss >= 0f);
        Assert.False(float.IsNaN(loss) || float.IsInfinity(loss));
    }

    [Fact]
    public void Run_InvokesOnStepForEveryStepWithIncreasingIndex()
    {
        var (model, sampler) = BuildTinySetup(new Random(1));
        var optimizer = new SgdOptimizer(model.Parameters(), learningRate: 0.01f);
        var trainer = new Trainer(model, sampler, optimizer);

        var seenSteps = new List<int>();
        trainer.Run(steps: 5, batchSize: 2, onStep: (step, loss) => seenSteps.Add(step));

        Assert.Equal([0, 1, 2, 3, 4], seenSteps);
    }

    [Fact]
    public void Run_LossDecreasesOnARepetitiveCorpus()
    {
        // The real end-to-end proof: GptModel + BatchSampler +
        // CrossEntropyLoss + an optimizer, wired together by Trainer,
        // should actually learn something - loss on a trivially-repetitive
        // 4-token-cycle corpus should drop substantially within a modest
        // number of steps.
        var (model, sampler) = BuildTinySetup(new Random(42));
        var optimizer = new AdamWOptimizer(model.Parameters(), learningRate: 0.01f, weightDecay: 0f);
        var trainer = new Trainer(model, sampler, optimizer);

        float firstLoss = trainer.Step(batchSize: 8);

        float lastLoss = 0f;
        for (int i = 0; i < 100; i++)
        {
            lastLoss = trainer.Step(batchSize: 8);
        }

        Assert.True(lastLoss < firstLoss * 0.5f, $"Expected loss to drop substantially: {firstLoss} -> {lastLoss}.");
    }
}
