namespace MiniChatGPT.Contracts;

public interface ITokenizer : IContractFingerprint
{
    int VocabSize { get; }
    int[] Encode(string text);
    string Decode(ReadOnlySpan<int> tokens);
    object GetPayloadForCheckpoint();
}

public interface ILanguageModel : IContractFingerprint
{
    string ModelKind { get; }
    int VocabSize { get; }
    float[] NextTokenScores(ReadOnlySpan<int> context);
    object GetPayloadForCheckpoint();
}

public interface ITextGenerator
{
    string Generate(string prompt, int maxTokens, float temperature, int topK, int? seed = null);
}

public interface ICheckpointIO
{
    void Save(string path, Checkpoint checkpoint);
    Checkpoint Load(string path);
}

public sealed record Checkpoint(
    string ModelKind,
    string TokenizerKind,
    object TokenizerPayload,
    object ModelPayload,
    int Seed,
    string ContractFingerprintChain
);

public interface IContractFingerprint
{
    string GetContractFingerprint();
}
