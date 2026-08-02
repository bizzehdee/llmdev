using TensorValue = Tensor.Tensor;

namespace Model;

/// <summary>
/// Shared weight-initialisation helper: small Gaussian noise (mean 0,
/// caller-chosen std - GPT-2 uses 0.02 for embeddings) rather than zeros,
/// since identical rows/weights would otherwise get identical gradients
/// forever and never differentiate from each other.
/// </summary>
internal static class GaussianInit
{
    public static TensorValue Matrix(int rows, int cols, float stdDev, Random random)
    {
        var values = new float[rows * cols];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Sample(random) * stdDev;
        }
        return TensorValue.FromValues(values, [rows, cols]);
    }

    /// <summary>
    /// Standard normal sample via the Box-Muller transform - .NET's Random
    /// only gives uniform samples, and this is a from-first-principles
    /// project, so no external distribution library either.
    /// </summary>
    public static float Sample(Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
    }
}
