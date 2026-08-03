using Tensor;
using Xunit;
using TensorValue = Tensor.Tensor;

namespace Training.Tests;

public class CrossEntropyLossTests
{
    [Fact]
    public void Compute_ReturnsScalarShape()
    {
        var logits = new Variable(TensorValue.Zeros([3, 5]));

        var loss = CrossEntropyLoss.Compute(logits, [0, 1, 2]);

        Assert.Equal(new[] { 1 }, loss.Value.Shape);
    }

    [Fact]
    public void Compute_UniformLogits_EqualsLogVocabSize()
    {
        // softmax of all-zero logits is uniform (1/vocabSize everywhere), so
        // -log(1/vocabSize) = log(vocabSize) exactly, for every position -
        // an exact expected value rather than just a plausibility check.
        const int vocabSize = 8;
        var logits = new Variable(TensorValue.Zeros([4, vocabSize]));

        var loss = CrossEntropyLoss.Compute(logits, [0, 3, 7, 1]);

        Assert.Equal(MathF.Log(vocabSize), loss.Value.ToArray()[0], precision: 4);
    }

    [Fact]
    public void Compute_ConfidentCorrectPrediction_HasLowLoss()
    {
        var values = new float[] { 20, -20, -20 }; // overwhelmingly favours index 0
        var logits = new Variable(TensorValue.FromValues(values, [1, 3]));

        var loss = CrossEntropyLoss.Compute(logits, [0]);

        Assert.True(loss.Value.ToArray()[0] < 1e-3f, $"Expected near-zero loss, got {loss.Value.ToArray()[0]}.");
    }

    [Fact]
    public void Compute_ConfidentWrongPrediction_HasHighLoss()
    {
        var values = new float[] { 20, -20, -20 }; // overwhelmingly favours index 0, but target is index 1
        var logits = new Variable(TensorValue.FromValues(values, [1, 3]));

        var loss = CrossEntropyLoss.Compute(logits, [1]);

        Assert.True(loss.Value.ToArray()[0] > 10f, $"Expected a large loss, got {loss.Value.ToArray()[0]}.");
    }

    [Fact]
    public void Compute_GradientPushesTargetLogitUpAndOthersDown()
    {
        var logits = new Variable(TensorValue.Zeros([1, 3]));

        var loss = CrossEntropyLoss.Compute(logits, [1]);
        loss.Backward();

        var grad = logits.Gradient.ToArray();
        // Gradient descent moves logits by -grad, so a negative gradient at
        // the target index means the target logit will increase.
        Assert.True(grad[1] < 0f, "Expected the target logit's gradient to be negative (pushing it up).");
        Assert.True(grad[0] > 0f, "Expected a non-target logit's gradient to be positive (pushing it down).");
        Assert.True(grad[2] > 0f, "Expected a non-target logit's gradient to be positive (pushing it down).");
    }

    [Fact]
    public void Compute_LargeMagnitudeLogits_StaysFiniteAndMatchesShiftedEquivalent()
    {
        // TASK-017: without the log-sum-exp shift, exp(1000) overflows to
        // +Infinity and the loss becomes NaN. Shifting every logit in a row
        // by the same constant shouldn't change the loss at all.
        var values = new float[] { 1000, 1001, 1002 };
        var shiftedValues = new float[] { 0, 1, 2 };

        var loss = CrossEntropyLoss.Compute(new Variable(TensorValue.FromValues(values, [1, 3])), [1]);
        var shiftedLoss = CrossEntropyLoss.Compute(new Variable(TensorValue.FromValues(shiftedValues, [1, 3])), [1]);

        Assert.True(float.IsFinite(loss.Value.ToArray()[0]));
        Assert.Equal(shiftedLoss.Value.ToArray()[0], loss.Value.ToArray()[0], precision: 4);
    }

    [Fact]
    public void Compute_GradientMatchesFiniteDifference()
    {
        var rng = new Random(7);
        var values = new float[4 * 5];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        var logits = new Variable(TensorValue.FromValues(values, [4, 5]));
        int[] targets = [1, 3, 0, 4];

        logits.ZeroGrad();
        CrossEntropyLoss.Compute(logits, targets).Backward();
        var analytic = logits.Gradient.ToArray();

        const float epsilon = 1e-3f;
        for (int i = 0; i < values.Length; i++)
        {
            var plus = (float[])values.Clone();
            plus[i] += epsilon;
            float lossPlus = CrossEntropyLoss.Compute(new Variable(TensorValue.FromValues(plus, [4, 5])), targets).Value.ToArray()[0];

            var minus = (float[])values.Clone();
            minus[i] -= epsilon;
            float lossMinus = CrossEntropyLoss.Compute(new Variable(TensorValue.FromValues(minus, [4, 5])), targets).Value.ToArray()[0];

            float numeric = (lossPlus - lossMinus) / (2 * epsilon);
            Assert.True(MathF.Abs(numeric - analytic[i]) < 5e-2f, $"Gradient mismatch at {i}: analytic={analytic[i]}, numeric={numeric}");
        }
    }
}
