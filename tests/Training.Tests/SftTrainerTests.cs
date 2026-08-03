using Model;
using Xunit;

namespace Training.Tests;

public class SftTrainerTests
{
    private static IReadOnlyList<SftTokenizedExample> BuildTinyExamples()
    {
        // A single, trivially learnable pattern repeated across several
        // "examples" - a tiny model should be able to drive the loss on
        // the *masked* (response) positions down substantially.
        var example = new SftTokenizedExample(
            InputIds: [1, 2, 3],
            TargetIds: [2, 3, 4],
            ResponseMask: [false, true, true]);
        return Enumerable.Repeat(example, 8).ToList();
    }

    [Fact]
    public void Constructor_EmptyExampleListThrows()
    {
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(1));
        var optimizer = new SgdOptimizer(model.Parameters(), learningRate: 0.01f);

        Assert.Throws<ArgumentException>(() => new SftTrainer(model, [], optimizer));
    }

    [Fact]
    public void Step_ReturnsAFiniteNonNegativeLoss()
    {
        var examples = BuildTinyExamples();
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(1));
        var optimizer = new SgdOptimizer(model.Parameters(), learningRate: 0.01f);
        var trainer = new SftTrainer(model, examples, optimizer);

        float loss = trainer.Step(batchSize: 4, startIndex: 0);

        Assert.True(loss >= 0f);
        Assert.False(float.IsNaN(loss) || float.IsInfinity(loss));
    }

    [Fact]
    public void Step_WrapsAroundTheDatasetWhenBatchSizeExceedsExampleCount()
    {
        var examples = BuildTinyExamples();
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(1));
        var optimizer = new SgdOptimizer(model.Parameters(), learningRate: 0.01f);
        var trainer = new SftTrainer(model, examples, optimizer);

        float loss = trainer.Step(batchSize: examples.Count * 2 + 3, startIndex: 0);

        Assert.False(float.IsNaN(loss) || float.IsInfinity(loss));
    }

    [Fact]
    public void Run_InvokesOnStepForEveryStepWithIncreasingIndex()
    {
        var examples = BuildTinyExamples();
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(1));
        var optimizer = new SgdOptimizer(model.Parameters(), learningRate: 0.01f);
        var trainer = new SftTrainer(model, examples, optimizer);

        var seenSteps = new List<int>();
        trainer.Run(steps: 5, batchSize: 2, onStep: (step, loss) => seenSteps.Add(step));

        Assert.Equal([0, 1, 2, 3, 4], seenSteps);
    }

    [Fact]
    public void Run_LossDecreasesOnARepetitivePattern()
    {
        // The real end-to-end proof, mirroring TrainerTests' pretraining
        // equivalent: GptModel + masked CrossEntropyLoss + an optimizer,
        // wired together by SftTrainer, should actually learn something.
        var examples = BuildTinyExamples();
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(42));
        var optimizer = new AdamWOptimizer(model.Parameters(), learningRate: 0.01f, weightDecay: 0f);
        var trainer = new SftTrainer(model, examples, optimizer);

        float firstLoss = trainer.Step(batchSize: 8, startIndex: 0);

        float lastLoss = 0f;
        for (int i = 0; i < 100; i++)
        {
            lastLoss = trainer.Step(batchSize: 8, startIndex: 0);
        }

        Assert.True(lastLoss < firstLoss * 0.5f, $"Expected loss to drop substantially: {firstLoss} -> {lastLoss}.");
    }

    // TASK-030: epoch-based training.

    [Fact]
    public void RunEpochs_InvokesOnStepExactlyCeilOfDatasetSizeOverBatchSizeTimesEpochs()
    {
        var examples = BuildTinyExamples(); // 8 examples
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(1));
        var optimizer = new SgdOptimizer(model.Parameters(), learningRate: 0.01f);
        var trainer = new SftTrainer(model, examples, optimizer);

        var seenGlobalSteps = new List<int>();
        var seenEpochs = new List<int>();
        trainer.RunEpochs(epochs: 3, batchSize: 3, random: new Random(1), onStep: (epoch, globalStep, loss) =>
        {
            seenEpochs.Add(epoch);
            seenGlobalSteps.Add(globalStep);
        });

        // ceil(8 / 3) = 3 steps/epoch * 3 epochs = 9 steps total.
        Assert.Equal(Enumerable.Range(0, 9).ToList(), seenGlobalSteps);
        Assert.Equal([0, 0, 0, 1, 1, 1, 2, 2, 2], seenEpochs);
    }

    [Fact]
    public void RunEpochs_LastBatchOfAnEpochIsSmallerWhenDatasetSizeIsNotAMultipleOfBatchSize()
    {
        var examples = BuildTinyExamples(); // 8 examples
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(1));
        var optimizer = new SgdOptimizer(model.Parameters(), learningRate: 0.01f);
        var trainer = new SftTrainer(model, examples, optimizer);

        // A batch size that doesn't evenly divide the dataset shouldn't
        // throw or skip examples - the final batch of each epoch just runs
        // smaller (StepOn averages by actual batch size, not a fixed one).
        int stepCount = 0;
        trainer.RunEpochs(epochs: 1, batchSize: 5, random: new Random(1), onStep: (epoch, globalStep, loss) =>
        {
            Assert.False(float.IsNaN(loss) || float.IsInfinity(loss));
            stepCount++;
        });

        Assert.Equal(2, stepCount); // ceil(8 / 5) = 2 steps: a batch of 5, then a batch of 3.
    }

    [Fact]
    public void RunEpochs_LossDecreasesOnARepetitivePattern()
    {
        var examples = BuildTinyExamples();
        var model = new GptModel(vocabSize: 6, embeddingDim: 8, numLayers: 1, numHeads: 2, maxSequenceLength: 4, random: new Random(42));
        var optimizer = new AdamWOptimizer(model.Parameters(), learningRate: 0.01f, weightDecay: 0f);
        var trainer = new SftTrainer(model, examples, optimizer);

        var losses = new List<float>();
        trainer.RunEpochs(epochs: 30, batchSize: 8, random: new Random(7), onStep: (epoch, globalStep, loss) => losses.Add(loss));

        Assert.True(losses[^1] < losses[0] * 0.5f, $"Expected loss to drop substantially: {losses[0]} -> {losses[^1]}.");
    }
}
