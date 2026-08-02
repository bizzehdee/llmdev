using Tensor;
using Xunit;
using TensorValue = Tensor.Tensor;

namespace Training.Tests;

public class OptimizerTests
{
    [Fact]
    public void Sgd_Step_MovesParameterByLearningRateTimesGradient()
    {
        var parameter = new Variable(TensorValue.FromValues([10f], [1]));
        parameter.Backward(TensorValue.FromValues([4f], [1])); // seed an arbitrary gradient
        var sgd = new SgdOptimizer([parameter], learningRate: 0.5f);

        sgd.Step();

        // 10 - 0.5*4 = 8
        Assert.Equal(8f, parameter.Value.ToArray()[0], precision: 5);
    }

    [Fact]
    public void Sgd_ZeroGrad_ResetsGradientToZero()
    {
        var parameter = new Variable(TensorValue.FromValues([10f], [1]));
        parameter.Backward(TensorValue.FromValues([4f], [1]));
        var sgd = new SgdOptimizer([parameter], learningRate: 0.5f);

        sgd.ZeroGrad();

        Assert.Equal(0f, parameter.Gradient.ToArray()[0]);
    }

    [Fact]
    public void Sgd_MinimisesASimpleQuadratic()
    {
        // Minimise (x - 5)^2 via repeated gradient steps; x should converge towards 5.
        var x = new Variable(TensorValue.FromValues([0f], [1]));
        var target = new Variable(TensorValue.FromValues([5f], [1]));
        var sgd = new SgdOptimizer([x], learningRate: 0.1f);

        for (int step = 0; step < 200; step++)
        {
            sgd.ZeroGrad();
            var diff = x.Subtract(target);
            var loss = diff.Multiply(diff);
            loss.Backward();
            sgd.Step();
        }

        Assert.Equal(5f, x.Value.ToArray()[0], precision: 2);
    }

    [Fact]
    public void AdamW_FirstStep_MatchesHandComputedUpdate()
    {
        // First step, m0=v0=0: m1 = (1-beta1)*g, v1 = (1-beta2)*g^2, bias
        // correction cancels the (1-beta) factors exactly on step 1 (mHat =
        // g, vHat = g^2), so the update simplifies to a known closed form:
        // w -= lr * (g / (|g| + eps) + weightDecay * w).
        const float lr = 0.1f, weightDecay = 0.01f, eps = 1e-8f, g = 4f, w0 = 10f;
        var parameter = new Variable(TensorValue.FromValues([w0], [1]));
        parameter.Backward(TensorValue.FromValues([g], [1]));
        var adamW = new AdamWOptimizer([parameter], learningRate: lr, weightDecay: weightDecay, epsilon: eps);

        adamW.Step();

        float expected = w0 - lr * (g / (MathF.Abs(g) + eps) + weightDecay * w0);
        Assert.Equal(expected, parameter.Value.ToArray()[0], precision: 4);
    }

    [Fact]
    public void AdamW_ZeroGrad_ResetsGradientToZero()
    {
        var parameter = new Variable(TensorValue.FromValues([10f], [1]));
        parameter.Backward(TensorValue.FromValues([4f], [1]));
        var adamW = new AdamWOptimizer([parameter], learningRate: 0.1f);

        adamW.ZeroGrad();

        Assert.Equal(0f, parameter.Gradient.ToArray()[0]);
    }

    [Fact]
    public void AdamW_MinimisesASimpleQuadratic()
    {
        var x = new Variable(TensorValue.FromValues([0f], [1]));
        var target = new Variable(TensorValue.FromValues([5f], [1]));
        var adamW = new AdamWOptimizer([x], learningRate: 0.2f, weightDecay: 0f);

        for (int step = 0; step < 200; step++)
        {
            adamW.ZeroGrad();
            var diff = x.Subtract(target);
            var loss = diff.Multiply(diff);
            loss.Backward();
            adamW.Step();
        }

        Assert.Equal(5f, x.Value.ToArray()[0], precision: 1);
    }

    [Fact]
    public void AdamW_TracksMultipleParametersIndependently()
    {
        var a = new Variable(TensorValue.FromValues([1f], [1]));
        var b = new Variable(TensorValue.FromValues([1f], [1]));
        var adamW = new AdamWOptimizer([a, b], learningRate: 0.1f);

        // Opposite-sign gradients: on Adam's very first step, the adaptive
        // step size is normalised by the gradient's own magnitude (mHat/vHat
        // both reduce to g/|g| when the moment estimates start at zero), so
        // *same*-signed gradients of different magnitude would misleadingly
        // produce near-identical updates on step one - not a sign of a bug.
        // Opposite signs isolate what actually matters here: each
        // parameter's moment state must not leak into the other's update.
        a.Backward(TensorValue.FromValues([10f], [1]));
        b.Backward(TensorValue.FromValues([-10f], [1]));
        adamW.Step();

        Assert.True(a.Value.ToArray()[0] < 1f, "Expected a positive gradient to decrease its parameter.");
        Assert.True(b.Value.ToArray()[0] > 1f, "Expected a negative gradient to increase its parameter.");
    }

    [Fact]
    public void TrainingLoop_CrossEntropyLossDecreasesOverSteps()
    {
        // End-to-end sanity check of the whole TASK-011 stack together:
        // treat a small logits matrix as directly-trainable "predictions"
        // (standing in for what would normally be a model's output) and
        // confirm optimizing them against a fixed set of targets via
        // CrossEntropyLoss + AdamW actually drives the loss down.
        var logits = new Variable(TensorValue.Zeros([3, 5]));
        int[] targets = [1, 3, 0];
        var adamW = new AdamWOptimizer([logits], learningRate: 0.3f, weightDecay: 0f);

        float initialLoss = CrossEntropyLoss.Compute(logits, targets).Value.ToArray()[0];

        for (int step = 0; step < 50; step++)
        {
            adamW.ZeroGrad();
            var loss = CrossEntropyLoss.Compute(logits, targets);
            loss.Backward();
            adamW.Step();
        }

        float finalLoss = CrossEntropyLoss.Compute(logits, targets).Value.ToArray()[0];
        Assert.True(finalLoss < initialLoss * 0.1f, $"Expected loss to drop sharply: {initialLoss} -> {finalLoss}.");
    }
}
