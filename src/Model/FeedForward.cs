using Tensor;
using TensorValue = Tensor.Tensor;

namespace Model;

/// <summary>
/// The position-wise feed-forward network in a transformer block: expand
/// to a wider hidden dimension, apply a non-linearity (ReLU - see
/// PLAN.md/TASK-004 on choosing it over GELU), project back down. Applied
/// independently to every position (no mixing across positions - that's
/// attention's job), giving the model per-position computation capacity on
/// top of what attention gathers.
/// </summary>
public sealed class FeedForward
{
    private const float InitStdDev = 0.02f; // GPT-2's init scale

    public int EmbeddingDim { get; }
    public int HiddenDim { get; }

    public Variable InputWeight { get; }
    public Variable InputBias { get; }
    public Variable OutputWeight { get; }
    public Variable OutputBias { get; }

    public FeedForward(int embeddingDim, int hiddenDim, Random? random = null)
    {
        EmbeddingDim = embeddingDim;
        HiddenDim = hiddenDim;

        random ??= new Random();
        InputWeight = new Variable(GaussianInit.Matrix(embeddingDim, hiddenDim, InitStdDev, random));
        InputBias = new Variable(TensorValue.Zeros([hiddenDim]));
        OutputWeight = new Variable(GaussianInit.Matrix(hiddenDim, embeddingDim, InitStdDev, random));
        OutputBias = new Variable(TensorValue.Zeros([embeddingDim]));
    }

    public Variable Forward(Variable x)
    {
        var hidden = x.MatMul(InputWeight).Add(InputBias).Relu();
        return hidden.MatMul(OutputWeight).Add(OutputBias);
    }
}
