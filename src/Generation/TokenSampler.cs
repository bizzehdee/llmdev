namespace Generation;

/// <summary>
/// Turns a row of raw logits into a chosen token id, per
/// <see cref="SamplingOptions"/>. Deliberately plain float-array math, not
/// Tensor/Variable: sampling is an inference-only, no-gradient operation,
/// so building an autodiff graph for it would be pure waste.
/// </summary>
public static class TokenSampler
{
    public static int Sample(float[] logits, SamplingOptions options, Random random)
    {
        if (options.Temperature <= 0f)
        {
            return ArgMax(logits);
        }

        var scaled = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            scaled[i] = logits[i] / options.Temperature;
        }

        if (options.TopK is int topK)
        {
            scaled = ApplyTopK(scaled, topK);
        }
        if (options.TopP is float topP)
        {
            scaled = ApplyTopP(scaled, topP);
        }

        var probabilities = Softmax(scaled);
        return SampleFromDistribution(probabilities, random);
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }
        return best;
    }

    /// <summary>
    /// Softmax with the max-subtraction numerical-stability trick - unlike
    /// Variable.Softmax (see TASK-004), this is plain non-differentiable
    /// float math with no backward pass to keep simple, so there's no
    /// reason not to include it, and temperature scaling can push logits to
    /// extremes where it genuinely matters.
    /// </summary>
    private static float[] Softmax(float[] logits)
    {
        float max = float.NegativeInfinity;
        foreach (float v in logits)
        {
            if (v > max)
            {
                max = v;
            }
        }

        var exp = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            exp[i] = MathF.Exp(logits[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < exp.Length; i++)
        {
            exp[i] /= sum;
        }
        return exp;
    }

    private static float[] ApplyTopK(float[] logits, int k)
    {
        if (k <= 0 || k >= logits.Length)
        {
            return logits;
        }

        float threshold = logits.OrderByDescending(x => x).Take(k).Min();
        var result = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = logits[i] >= threshold ? logits[i] : float.NegativeInfinity;
        }
        return result;
    }

    private static float[] ApplyTopP(float[] logits, float p)
    {
        var probabilities = Softmax(logits);
        var order = Enumerable.Range(0, logits.Length).OrderByDescending(i => probabilities[i]);

        var kept = new HashSet<int>();
        float cumulative = 0f;
        foreach (int index in order)
        {
            kept.Add(index);
            cumulative += probabilities[index];
            if (cumulative >= p)
            {
                break;
            }
        }

        var result = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = kept.Contains(i) ? logits[i] : float.NegativeInfinity;
        }
        return result;
    }

    private static int SampleFromDistribution(float[] probabilities, Random random)
    {
        float draw = (float)random.NextDouble();
        float cumulative = 0f;
        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (draw <= cumulative)
            {
                return i;
            }
        }
        return probabilities.Length - 1; // floating-point rounding fallback
    }
}
