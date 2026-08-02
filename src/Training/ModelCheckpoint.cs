using Model;

namespace Training;

/// <summary>
/// Saves/loads a <see cref="GptModel"/>'s architecture and learned weights
/// to/from a binary file. Loading reconstructs a fresh model from the saved
/// hyperparameters (so its parameter shapes are guaranteed to match, by
/// construction) and then overwrites every parameter's values in place -
/// see Tensor.LoadInPlace.
/// </summary>
public static class ModelCheckpoint
{
    public static void Save(GptModel model, string path)
    {
        using var writer = new BinaryWriter(File.Create(path));

        writer.Write(model.VocabSize);
        writer.Write(model.EmbeddingDim);
        writer.Write(model.NumLayers);
        writer.Write(model.NumHeads);
        writer.Write(model.MaxSequenceLength);
        writer.Write(model.Blocks[0].FeedForward.HiddenDim);

        var parameters = model.Parameters();
        writer.Write(parameters.Count);
        foreach (var parameter in parameters)
        {
            var shape = parameter.Value.Shape;
            writer.Write(shape.Length);
            foreach (int dim in shape)
            {
                writer.Write(dim);
            }
            foreach (float value in parameter.Value.ToArray())
            {
                writer.Write(value);
            }
        }
    }

    public static GptModel Load(string path, Random? random = null)
    {
        using var reader = new BinaryReader(File.OpenRead(path));

        int vocabSize = reader.ReadInt32();
        int embeddingDim = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int maxSequenceLength = reader.ReadInt32();
        int feedForwardHiddenDim = reader.ReadInt32();

        var model = new GptModel(vocabSize, embeddingDim, numLayers, numHeads, maxSequenceLength, feedForwardHiddenDim, random ?? new Random());
        var parameters = model.Parameters();

        int savedParameterCount = reader.ReadInt32();
        if (savedParameterCount != parameters.Count)
        {
            throw new InvalidOperationException($"Checkpoint has {savedParameterCount} parameters, but a model with this architecture has {parameters.Count}.");
        }

        foreach (var parameter in parameters)
        {
            int rank = reader.ReadInt32();
            var shape = new int[rank];
            for (int i = 0; i < rank; i++)
            {
                shape[i] = reader.ReadInt32();
            }
            if (!shape.SequenceEqual(parameter.Value.Shape))
            {
                throw new InvalidOperationException($"Checkpoint parameter shape [{string.Join(",", shape)}] doesn't match the reconstructed model's [{string.Join(",", parameter.Value.Shape)}].");
            }

            var values = new float[parameter.Value.Length];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = reader.ReadSingle();
            }
            parameter.Value.LoadInPlace(values);
        }

        return model;
    }
}
