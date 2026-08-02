namespace Training;

/// <summary>
/// One next-token-prediction training example: <see cref="Target"/> is
/// <see cref="Input"/> shifted one position later in the corpus, so
/// Target[i] is the correct next token after Input[0..i].
/// </summary>
public readonly record struct TrainingExample(int[] Input, int[] Target);
