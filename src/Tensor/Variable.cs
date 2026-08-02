namespace Tensor;

/// <summary>
/// A node in a reverse-mode automatic differentiation graph: wraps a
/// <see cref="Tensor"/> value, its accumulated gradient, and (for the
/// result of an op) a closure that knows how to push that gradient back to
/// its inputs. Every op in the Variable.*.cs files follows the same shape:
/// compute the forward <see cref="Tensor"/> value using the plain Tensor
/// ops from TASK-003, then call <see cref="FromOp"/> with a factory that -
/// given the (not-yet-existing-until-now) result variable - builds a
/// closure reading *its* <see cref="Gradient"/> (assumed already fully
/// accumulated by the time it runs - see <see cref="Backward"/>) and adding
/// each input's share of it to that input's own <see cref="Gradient"/> via
/// <see cref="AccumulateGradient"/>.
///
/// This mirrors the classic micrograd design: a topological sort of the
/// graph ending at the node <see cref="Backward"/> was called on, then
/// walking that order in reverse so a node's gradient is only consumed
/// once every path that flows through it has already contributed.
/// </summary>
public sealed partial class Variable
{
    private readonly List<Variable> _parents = new();
    private Action? _backwardFn;

    public Tensor Value { get; }
    public Tensor Gradient { get; private set; }

    public Variable(Tensor value)
    {
        Value = value;
        Gradient = Tensor.Zeros(value.Shape);
    }

    /// <summary>
    /// Builds the output variable of an op. <paramref name="parents"/> are
    /// the inputs the op read; <paramref name="backwardFnFactory"/> is
    /// given the new output variable (so its closure can read *that*
    /// variable's accumulated gradient - not any parent's) and must return
    /// a closure that adds each parent's share of it to the parent's own
    /// gradient.
    /// </summary>
    internal static Variable FromOp(Tensor value, Func<Variable, Action> backwardFnFactory, params Variable[] parents)
    {
        var result = new Variable(value);
        result._backwardFn = backwardFnFactory(result);
        result._parents.AddRange(parents);
        return result;
    }

    internal void AccumulateGradient(Tensor contribution)
    {
        Gradient = Gradient.Add(contribution);
    }

    /// <summary>
    /// Seeds this variable's gradient (1, broadcast to its shape, if none
    /// is given - the standard "gradient of a scalar loss w.r.t. itself is
    /// 1" starting point) and propagates gradients back through every
    /// variable that contributed to it.
    /// </summary>
    public void Backward(Tensor? seed = null)
    {
        Gradient = seed ?? Tensor.FromValues(Enumerable.Repeat(1f, Value.Length).ToArray(), Value.Shape);

        var topoOrder = new List<Variable>();
        var visited = new HashSet<Variable>();

        void Visit(Variable v)
        {
            if (!visited.Add(v))
            {
                return;
            }
            foreach (var parent in v._parents)
            {
                Visit(parent);
            }
            topoOrder.Add(v);
        }

        Visit(this);
        topoOrder.Reverse();

        foreach (var v in topoOrder)
        {
            v._backwardFn?.Invoke();
        }
    }

    /// <summary>Resets this variable's gradient to zero, e.g. between training steps.</summary>
    public void ZeroGrad() => Gradient = Tensor.Zeros(Value.Shape);
}
