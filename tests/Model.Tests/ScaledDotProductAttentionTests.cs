using Tensor;
using Xunit;
using TensorValue = Tensor.Tensor;

namespace Model.Tests;

public class ScaledDotProductAttentionTests
{
    [Fact]
    public void Compute_OutputShapeMatchesValueShape()
    {
        var q = RandomVariable([4, 8]);
        var k = RandomVariable([4, 8]);
        var v = RandomVariable([4, 8]);

        var result = ScaledDotProductAttention.Compute(q, k, v, causal: false);

        Assert.Equal(new[] { 4, 8 }, result.Value.Shape);
    }

    [Fact]
    public void Compute_BatchedOverHeadsShapeMatchesValueShape()
    {
        var q = RandomVariable([3, 4, 8]); // [numHeads, seqLen, headDim]
        var k = RandomVariable([3, 4, 8]);
        var v = RandomVariable([3, 4, 8]);

        var result = ScaledDotProductAttention.Compute(q, k, v, causal: true);

        Assert.Equal(new[] { 3, 4, 8 }, result.Value.Shape);
    }

    [Fact]
    public void Compute_IdenticalValueRowsProduceThatRowRegardlessOfWeights()
    {
        // Attention output is a weighted average of V's rows; if every row of V is
        // identical, the weighted average must equal that row no matter what the
        // (unknown-to-us) attention weights turned out to be - a structural sanity
        // check that doesn't depend on reasoning about softmax's actual output.
        float[] row = [1f, 2f, 3f];
        var v = new Variable(TensorValue.FromValues([.. row, .. row, .. row], [3, 3]));
        var q = RandomVariable([3, 3]);
        var k = RandomVariable([3, 3]);

        var result = ScaledDotProductAttention.Compute(q, k, v, causal: false);

        for (int i = 0; i < 3; i++)
        {
            for (int d = 0; d < 3; d++)
            {
                Assert.Equal(row[d], result.Value[i, d], precision: 4);
            }
        }
    }

    [Fact]
    public void Compute_Causal_FuturePositionDoesNotAffectEarlierOutput()
    {
        var q = RandomVariable([4, 5]);
        var k = RandomVariable([4, 5]);
        var vBase = RandomVariable([4, 5]);
        var vChanged = CloneWithChangedRow(vBase, rowToChange: 3);

        var outputBase = ScaledDotProductAttention.Compute(q, k, vBase, causal: true);
        var outputChanged = ScaledDotProductAttention.Compute(q, k, vChanged, causal: true);

        for (int i = 0; i < 3; i++) // positions 0..2 must be unaffected by position 3 changing
        {
            for (int d = 0; d < 5; d++)
            {
                Assert.Equal(outputBase.Value[i, d], outputChanged.Value[i, d], precision: 4);
            }
        }
    }

    [Fact]
    public void Compute_NonCausal_FuturePositionCanAffectEarlierOutput()
    {
        var q = RandomVariable([4, 5]);
        var k = RandomVariable([4, 5]);
        var vBase = RandomVariable([4, 5]);
        var vChanged = CloneWithChangedRow(vBase, rowToChange: 3);

        var outputBase = ScaledDotProductAttention.Compute(q, k, vBase, causal: false);
        var outputChanged = ScaledDotProductAttention.Compute(q, k, vChanged, causal: false);

        bool anyEarlierPositionChanged = false;
        for (int i = 0; i < 3; i++)
        {
            for (int d = 0; d < 5; d++)
            {
                if (MathF.Abs(outputBase.Value[i, d] - outputChanged.Value[i, d]) > 1e-4f)
                {
                    anyEarlierPositionChanged = true;
                }
            }
        }
        Assert.True(anyEarlierPositionChanged, "Expected non-causal attention to let an earlier position's output depend on a later position's value.");
    }

    [Fact]
    public void Compute_AttentionWeightsSumToOne()
    {
        // Indirect check (Compute doesn't expose weights directly): weighted
        // average of a one-hot-per-dimension basis reveals the weights themselves.
        var q = RandomVariable([3, 4]);
        var k = RandomVariable([3, 4]);
        var identity = TensorValue.FromValues([1, 0, 0, 0, 1, 0, 0, 0, 1], [3, 3]);
        var v = new Variable(identity);

        var result = ScaledDotProductAttention.Compute(q, k, v, causal: false);

        for (int i = 0; i < 3; i++)
        {
            float rowSum = result.Value[i, 0] + result.Value[i, 1] + result.Value[i, 2];
            Assert.Equal(1f, rowSum, precision: 4);
        }
    }

    [Fact]
    public void Compute_QueryOffset_SingleNewQueryAttendsToEveryPriorKey()
    {
        // TASK-020's KV-cache case: one new query row (absolute position
        // queryOffset) against every key up to and including its own -
        // since queryOffset + 0 >= every key index < keyLen, nothing
        // should be masked at all.
        const int keyLen = 5, queryOffset = 4;
        var q = RandomVariable([1, 4]);
        var k = RandomVariable([keyLen, 4]);
        var v = RandomVariable([keyLen, 4]);

        var causalResult = ScaledDotProductAttention.Compute(q, k, v, causal: true, queryOffset);
        var nonCausalResult = ScaledDotProductAttention.Compute(q, k, v, causal: false);

        for (int d = 0; d < 4; d++)
        {
            Assert.Equal(nonCausalResult.Value[0, d], causalResult.Value[0, d], precision: 4);
        }
    }

    [Fact]
    public void Compute_QueryOffset_MatchesEquivalentFullSquareMask()
    {
        // A rectangular mask with queryOffset == keyLen - queryLen should
        // produce exactly the rows a full square causal mask over the
        // whole sequence would, for those same rows - the property the
        // KV-cache path's correctness rests on.
        const int fullLen = 5, newLen = 2;
        var fullQ = RandomVariable([fullLen, 4]);
        var k = RandomVariable([fullLen, 4]);
        var v = RandomVariable([fullLen, 4]);

        var fullResult = ScaledDotProductAttention.Compute(fullQ, k, v, causal: true);

        var newQValues = fullQ.Value.ToArray().Skip((fullLen - newLen) * 4).ToArray();
        var newQ = new Variable(TensorValue.FromValues(newQValues, [newLen, 4]));
        var incrementalResult = ScaledDotProductAttention.Compute(newQ, k, v, causal: true, queryOffset: fullLen - newLen);

        for (int i = 0; i < newLen; i++)
        {
            for (int d = 0; d < 4; d++)
            {
                Assert.Equal(fullResult.Value[fullLen - newLen + i, d], incrementalResult.Value[i, d], precision: 4);
            }
        }
    }

    [Fact]
    public void Compute_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => ScaledDotProductAttention.Compute(vars[0], vars[1], vars[2], causal: false),
            RandomVariable([3, 4]), RandomVariable([3, 4]), RandomVariable([3, 4]));
    }

    [Fact]
    public void Compute_Causal_GradientMatchesFiniteDifference()
    {
        CheckGradient(vars => ScaledDotProductAttention.Compute(vars[0], vars[1], vars[2], causal: true),
            RandomVariable([3, 4]), RandomVariable([3, 4]), RandomVariable([3, 4]));
    }

    private static Variable CloneWithChangedRow(Variable source, int rowToChange)
    {
        var values = source.Value.ToArray();
        int cols = source.Value.Shape[1];
        for (int d = 0; d < cols; d++)
        {
            values[rowToChange * cols + d] += 10f;
        }
        return new Variable(TensorValue.FromValues(values, source.Value.Shape));
    }

    private static readonly Random Rng = new(2024);

    private static Variable RandomVariable(int[] shape, float min = -1f, float max = 1f)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = min + (float)Rng.NextDouble() * (max - min);
        }
        return new Variable(TensorValue.FromValues(values, shape));
    }

    private static void CheckGradient(Func<Variable[], Variable> forward, params Variable[] inputs)
    {
        const float epsilon = 1e-3f;
        const float tolerance = 5e-2f;

        foreach (var v in inputs)
        {
            v.ZeroGrad();
        }
        var output = forward(inputs);
        output.Backward();

        for (int p = 0; p < inputs.Length; p++)
        {
            var original = inputs[p].Value.ToArray();
            var analytic = inputs[p].Gradient.ToArray();

            for (int i = 0; i < original.Length; i++)
            {
                var plusValues = (float[])original.Clone();
                plusValues[i] += epsilon;
                var plusInputs = (Variable[])inputs.Clone();
                plusInputs[p] = new Variable(TensorValue.FromValues(plusValues, inputs[p].Value.Shape));
                float sumPlus = forward(plusInputs).Value.ToArray().Sum();

                var minusValues = (float[])original.Clone();
                minusValues[i] -= epsilon;
                var minusInputs = (Variable[])inputs.Clone();
                minusInputs[p] = new Variable(TensorValue.FromValues(minusValues, inputs[p].Value.Shape));
                float sumMinus = forward(minusInputs).Value.ToArray().Sum();

                float numeric = (sumPlus - sumMinus) / (2 * epsilon);

                Assert.True(
                    MathF.Abs(numeric - analytic[i]) < tolerance,
                    $"Gradient mismatch for input {p}, element {i}: analytic={analytic[i]}, numeric={numeric}");
            }
        }
    }
}
