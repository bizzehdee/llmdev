namespace Training;

/// <summary>
/// One raw instruction/response pair, as read from an SFT dataset file -
/// TASK-016's example-based data shape, distinct from <see cref="TokenCorpus"/>'s
/// continuous token stream: each pair here is trained as its own
/// standalone sequence, not a sliding window over one long stream.
/// </summary>
public sealed record SftExample(string Instruction, string Response);

/// <summary>
/// An <see cref="SftExample"/> after tokenisation via <see cref="SftDataset.Tokenize"/>:
/// the standard next-token-prediction shift (<see cref="TargetIds"/>[i] is
/// what should follow <see cref="InputIds"/>[i]), plus which target
/// positions actually fall within the response - the instruction/prompt
/// portion must not count towards the loss (see
/// <see cref="CrossEntropyLoss.ComputeMasked"/>).
/// </summary>
public sealed record SftTokenizedExample(int[] InputIds, int[] TargetIds, bool[] ResponseMask);
