using Xunit;

namespace Generation.Tests;

public class TokenSamplerTests
{
    [Fact]
    public void Sample_Greedy_AlwaysPicksHighestLogit()
    {
        float[] logits = [1f, 5f, 3f, -2f];
        var options = SamplingOptions.Greedy();

        for (int trial = 0; trial < 10; trial++)
        {
            int chosen = TokenSampler.Sample(logits, options, new Random(trial));
            Assert.Equal(1, chosen);
        }
    }

    [Fact]
    public void Sample_Greedy_IgnoresTopKAndTopP()
    {
        float[] logits = [1f, 5f, 3f, -2f];
        var options = new SamplingOptions { Temperature = 0f, TopK = 1, TopP = 0.01f };

        int chosen = TokenSampler.Sample(logits, options, new Random(1));

        Assert.Equal(1, chosen);
    }

    [Fact]
    public void Sample_TopKOne_BehavesLikeGreedy()
    {
        float[] logits = [1f, 5f, 3f, -2f];
        var options = new SamplingOptions { Temperature = 1f, TopK = 1 };

        for (int trial = 0; trial < 10; trial++)
        {
            int chosen = TokenSampler.Sample(logits, options, new Random(trial));
            Assert.Equal(1, chosen);
        }
    }

    [Fact]
    public void Sample_VerySmallTopP_AlmostAlwaysPicksHighestLogit()
    {
        float[] logits = [1f, 8f, 3f, -2f]; // index 1 dominates the softmax distribution
        var options = new SamplingOptions { Temperature = 1f, TopP = 0.01f };
        var random = new Random(1);

        int highestCount = 0;
        for (int trial = 0; trial < 100; trial++)
        {
            if (TokenSampler.Sample(logits, options, random) == 1)
            {
                highestCount++;
            }
        }

        Assert.True(highestCount >= 95, $"Expected the dominant token to be picked almost every time, got {highestCount}/100.");
    }

    [Fact]
    public void Sample_ResultIsAlwaysAValidIndex()
    {
        float[] logits = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f];
        var options = new SamplingOptions { Temperature = 1.5f };
        var random = new Random(1);

        for (int trial = 0; trial < 200; trial++)
        {
            int chosen = TokenSampler.Sample(logits, options, random);
            Assert.InRange(chosen, 0, logits.Length - 1);
        }
    }

    [Fact]
    public void Sample_EmpiricalFrequencyRoughlyMatchesSoftmaxProbability()
    {
        float[] logits = [1f, 2f, 3f]; // softmax ≈ [0.09, 0.244, 0.665]
        var options = new SamplingOptions { Temperature = 1f };
        var random = new Random(1);

        var counts = new int[3];
        const int trials = 20000;
        for (int i = 0; i < trials; i++)
        {
            counts[TokenSampler.Sample(logits, options, random)]++;
        }

        float[] expectedProbabilities = [0.09f, 0.244f, 0.665f];
        for (int i = 0; i < 3; i++)
        {
            float empirical = (float)counts[i] / trials;
            Assert.True(MathF.Abs(empirical - expectedProbabilities[i]) < 0.02f,
                $"Index {i}: expected ~{expectedProbabilities[i]}, got {empirical}.");
        }
    }

    [Fact]
    public void Sample_HighTemperature_FlattensTowardsUniform()
    {
        float[] logits = [1f, 10f]; // without flattening, index 1 would dominate almost completely
        var options = new SamplingOptions { Temperature = 100f };
        var random = new Random(1);

        var counts = new int[2];
        const int trials = 5000;
        for (int i = 0; i < trials; i++)
        {
            counts[TokenSampler.Sample(logits, options, random)]++;
        }

        float fractionIndex0 = (float)counts[0] / trials;
        Assert.True(fractionIndex0 > 0.35f, $"Expected a high temperature to make both outcomes roughly comparable, got fraction {fractionIndex0} for index 0.");
    }

    [Fact]
    public void Sample_TopKZeroOrLess_ActsAsUnrestricted()
    {
        float[] logits = [1f, 2f, 3f, 4f];
        var unrestricted = new SamplingOptions { Temperature = 1f };
        var topKZero = new SamplingOptions { Temperature = 1f, TopK = 0 };

        var counts = new int[4];
        const int trials = 5000;
        for (int i = 0; i < trials; i++)
        {
            counts[TokenSampler.Sample(logits, topKZero, new Random(i))]++;
        }

        // With TopK <= 0 treated as "no restriction," every index should still
        // be reachable, not just the single highest-logit one.
        Assert.True(counts.Count(c => c > 0) > 1, "Expected TopK=0 to not restrict sampling to a single token.");
    }

    [Fact]
    public void Sample_TopKAtOrAboveLength_ActsAsUnrestricted()
    {
        float[] logits = [1f, 2f, 3f, 4f];
        var options = new SamplingOptions { Temperature = 1f, TopK = 10 }; // >= logits.Length

        var counts = new int[4];
        const int trials = 5000;
        for (int i = 0; i < trials; i++)
        {
            counts[TokenSampler.Sample(logits, options, new Random(i))]++;
        }

        Assert.True(counts.Count(c => c > 0) > 1, "Expected TopK >= logits.Length to not restrict sampling to a single token.");
    }

    [Fact]
    public void Sample_TopP_CanRequireMultipleCandidatesBeforeReachingThreshold()
    {
        // Four near-equal-probability logits: reaching a high cumulative
        // threshold (0.95) requires accumulating several candidates, not
        // just the single top one - exercises the "haven't reached the
        // threshold yet, keep going" side of top-p's loop, not just the
        // "reached it, stop" side that a very small p always hits immediately.
        float[] logits = [1f, 1.01f, 0.99f, 1.02f];
        var options = new SamplingOptions { Temperature = 1f, TopP = 0.95f };
        var random = new Random(1);

        var counts = new int[4];
        const int trials = 2000;
        for (int i = 0; i < trials; i++)
        {
            counts[TokenSampler.Sample(logits, options, random)]++;
        }

        Assert.True(counts.Count(c => c > 0) >= 3, "Expected a high top-p threshold over near-equal logits to keep more than one or two candidates eligible.");
    }
}
