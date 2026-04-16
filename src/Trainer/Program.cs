using System;
using System.IO;
using System.Runtime.InteropServices.Marshalling;
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
            int batchSize = 1;
            int blockSize = 1;
            int checkpointInterval = epochs;

            if (args.Length < 1 || args[0] != "--train")
            {
                Console.WriteLine("Invalid command format.");
                Console.WriteLine("Type '--train -help' to see more information.");
                return;
            }

            if (args.Length >= 2 && args[1] == "-help")
            {
                PrintHelp();
                return;
            }

            for (int i = 1; i < args.Length; i++)
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
                    bool check = int.TryParse(args[++i], out epochs);
                    if (!check)
                    {
                        Console.WriteLine($"--epochs must be an integer.");
                        return;
                    }
                }
                else if (args[i] == "--out" && i + 1 < args.Length) 
                {
                    outPath = args[++i];
                }
                else if (args[i] == "--seed" && i + 1 < args.Length) 
                {
                    bool check = int.TryParse(args[++i], out seed);
                    if (!check)
                    {
                        Console.WriteLine($"--seed must be an integer.");
                        return;
                    }
                }
                else if (args[i] == "--lr" && i + 1 < args.Length) 
                {
                    bool check = float.TryParse(args[++i], out lr);
                    if (!check)
                    {
                        Console.WriteLine($"--lr must be a float.");
                        return;
                    }
                }
                else if (i + 1 < args.Length)
                {
                    Console.WriteLine($"Warning: Unknown option '{args[i]}' or argument list will be ignored.");
                    Console.WriteLine("Type '--train --help' to see more information.");
                }
                else if (args[i] == "--batch" && i + 1 < args.Length)
                {
                    batchSize = int.Parse(args[++i]);
                }
                else if (args[i] == "--block" && i + 1 < args.Length)
                {
                    blockSize = int.Parse(args[++i]);
                }
                else if (args[i] == "--interval" && i + 1 < args.Length)
                {
                    checkpointInterval = int.Parse(args[++i]);
                }
            }

            Console.WriteLine($"Training model {modelKind} on data {dataPath}...");

            CorpusLoader loader = new CorpusLoader(new DefaultFileSystem());
            Corpus corpus = loader.Load(dataPath, new CorpusLoadOptions(Lowercase: true));

            ITokenizer tokenizer = null;
            if (tokenizerKind == "char")
            {
                tokenizer = CharTokenizer.BuildFromText(corpus.TrainText);
            }
            else if (tokenizerKind == "word")
            {
                tokenizer = WordTokenizer.BuildFromText(corpus.TrainText);
            }
            else
            {
                Console.WriteLine($"Unknown tokenizer type: {tokenizerKind}");
                Console.WriteLine("Type '--train --help' to see more information.");
                return;
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
                    Console.WriteLine($"Unknown model type: {modelKind}");
                    Console.WriteLine("Type '--train --help' to see more information.");
                    return;
            }

            TrainingLoop trainingLoop = new TrainingLoop();
            TrainingConfig tConfig = new TrainingConfig(epochs, lr, checkpointInterval);      
            BatchConfig bConfig = new BatchConfig(batchSize, blockSize);
            
            ITokenStream stream = new ArrayTokenStream(tokens);
            IBatchProvider batchProvider = new TokenBatchProvider(stream, new BatchWindowSampler());

            try
            {
                var metrics = trainingLoop.Train(model, batchProvider, tConfig, bConfig, tokens, outPath);
                Console.WriteLine($"Training completed. Saving the checkpoint");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Training failed: {ex.Message}");
                return;
            }
            

            var checkpoint = new Checkpoint(
                ModelKind: modelKind,
                TokenizerKind: tokenizerKind,
                TokenizerPayload: tokenizer.GetPayloadForCheckpoint(),
                ModelPayload: model.GetPayloadForCheckpoint(),
                Seed: seed,
                ContractFingerprintChain: $"{loader.GetContractFingerprint()}|{tokenizer.GetContractFingerprint()}|{model.GetContractFingerprint()}"
            );

            try
            {
                JsonCheckpointIO checkpointIO = new JsonCheckpointIO();
                checkpointIO.Save(outPath, checkpoint);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Checkpoint save failed: {ex.Message}");
                return;
            }

            Console.WriteLine($"Model saved to file {outPath}");
        }

        static void PrintHelp()
        {
            Console.WriteLine("Usage: dotnet run –project src/Trainer --train [options]");
            Console.WriteLine("Options:");
            Console.WriteLine("  --data <path>         File path for training data (default: ../data/sample.txt)");
            Console.WriteLine("  --model <type>        Model type: bigram, trigram, tinynn, tinytransformer (default: trigram)");
            Console.WriteLine("  --tokenizer <type>    Tokenizer type: word, char (default: word)");
            Console.WriteLine("  --epochs <integer>    Quantity of training epochs (default: 3)");
            Console.WriteLine("  --out <path>          File path for checkpoint save (default: checkpoint.json)");
            Console.WriteLine("  --seed <integer>      Seed (default: 42)");
            Console.WriteLine("  --lr <float>          Learning rate for TinyNN (default: 0.1)");
            Console.WriteLine("   -help                See the list of available options");
        }
    }
}