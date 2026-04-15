using System;
using System.IO;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;
using Lib.Batching.Tests; 
using Lib.Corpus;
using Lib.Corpus.Configuration;
using Lib.Corpus.Infrastructure;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyTransformer.Factories;
using Lib.Tokenization.Application;
using Lib.Training;
using Lib.Training.Configuration;
using MiniChatGPT.Contracts; 

namespace Trainer
{
    class Program
    {
        static void Main(string[] args)
        {
            string dataPath = "../data/showcase.txt";
            string modelKind = "trigram";
            string tokenizerKind = "word";
            int epochs = 3;
            string outPath = "data/checkpoints/NGramCheckpoints.json";
            int seed = 42;
            float lr = 0.1f;

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--data" && i + 1 < args.Length) 
                {
                    dataPath = args[++i];
                }
                else if (args[i] == "--model" && i + 1 < args.Length) 
                {
                    modelKind = args[++i].ToLower();
                }
                else if (args[i] == "--tokenizer" && i + 1 < args.Length) 
                {
                    tokenizerKind = args[++i].ToLower();
                }
                else if (args[i] == "--epochs" && i + 1 < args.Length) 
                {
                    epochs = int.Parse(args[++i]);
                }
                else if (args[i] == "--out" && i + 1 < args.Length) 
                {
                    outPath = args[++i];
                }
                else if (args[i] == "--seed" && i + 1 < args.Length) 
                {
                    seed = int.Parse(args[++i]);
                }
                else if (args[i] == "--lr" && i + 1 < args.Length) 
                {
                    lr = float.Parse(args[++i]);
                }
            }

            Console.WriteLine($"Training model {modelKind} on data {dataPath}...");

            CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
            Corpus corpus = loader.Load(dataPath, new CorpusLoadOptions(Lowercase: true));

            ITokenizer tokenizer;
            if (tokenizerKind == "char")
            {
                tokenizer = CharTokenizer.BuildFromText(corpus.TrainText);
            }
            else
            {
                tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
            }

            int[] tokens = tokenizer.Encode(corpus.TrainText);

            ILanguageModel model;
            
            switch (modelKind)
            {
                case "bigram":
                    model = new NGramModelFactory().Create("bigram", tokenizer.VocabSize);
                    break;
                case "trigram":
                    model = new NGramModelFactory().Create("trigram", tokenizer.VocabSize);
                    break;
                case "tinynn":
                    model = new TinyNNModelFactory().CreateNewModel("tinynn", tokenizer.VocabSize);
                    break;
                case "tinytransformer":
                    model = new TinyTransformerModelFactory().Create(tokenizer.VocabSize, seed);
                    break;
                default:
                    throw new ArgumentException($"Unknown model type: {modelKind}");
            }

            TrainingLoop trainingLoop = new TrainingLoop();
            TrainingConfig tConfig = new TrainingConfig(epochs, lr, checkpointInterval: epochs);
            BatchConfig bConfig = new BatchConfig(BatchSize: 64, BlockSize: 32);
            
            ITokenStream stream = new ArrayTokenStream(tokens);
            IBatchProvider batchProvider = new TokenBatchProvider(stream, new BatchWindowSampler());

            var metrics = trainingLoop.Train(model, batchProvider, tConfig, bConfig, tokens, outPath);
            Console.WriteLine($"Training completed. Saving the checkpoint");

            var checkpoint = new Checkpoint(
                ModelKind: modelKind,
                TokenizerKind: tokenizerKind,
                TokenizerPayload: tokenizer.GetPayloadForCheckpoint(),
                ModelPayload: model.GetPayloadForCheckpoint(),
                Seed: seed,
                ContractFingerprintChain: $"{loader.GetContractFingerprint()}|{tokenizer.GetContractFingerprint()}|{model.GetContractFingerprint()}"
            );

            JsonCheckpointIO checkpointIO = new JsonCheckpointIO();
            checkpointIO.Save(outPath, checkpoint);

            Console.WriteLine($"Model saved to file {outPath}");
        }
    }
}