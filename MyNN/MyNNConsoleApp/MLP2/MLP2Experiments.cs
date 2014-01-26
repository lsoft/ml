using MyNN;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.OpenCL;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2Experiments
    {
        public static void Train()
        {
            int rndSeed = 123;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var mlp = new MLP(
                randomizer,
                null,
                null,
                new IFunction[]
                {
                    null,
                    new SigmoidFunction(1f),
                    new SigmoidFunction(1f),
                },
                new[]
                {
                    784,
                    400,
                    10
                }
                );

            var trainData = MNISTDataProvider.GetDataSet(
                "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                //"_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue,//1000,
                true);
            //trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                //"_MNIST_DATABASE/mnist/testset/",
                100,
                true);
            //validationData = validationData.ConvertToAutoencoder();

            var serialization = new SerializationHelper();

            //using (var clProvider = new CLProvider())
            //{
            //    var forward = new OpenCLForwardPropagation(
            //        mlp,
            //        clProvider);

            //    var outputList = forward.ComputeOutput(trainData);
            //    var stateList = forward.ComputeState(trainData);

            //    Console.WriteLine(outputList);
            //    Console.WriteLine(stateList);
            //}

            var config = new LearningAlgorithmConfig(
                new LinearLearningRate(0.1f, 0.99f),
                1,
                0.0f,
                1000,
                0.0001f,
                -1.0f);

            using (var clProvider = new CLProvider())
            {
                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    (processedMLP, processedConfig) => new OpenCLBackpropagationAlgorithm(
                        VectorizationSizeEnum.VectorizationMode16,
                        processedMLP,
                        processedConfig,
                        clProvider),
                    mlp,
                    new ClassificationValidation(
                        new FileSystemMLPSaver(serialization),
                        new HalfSquaredEuclidianDistance(), 
                        validationData,
                        300,
                        100), 
                    //new AutoencoderValidation(
                    //    validationData,
                    //    100,
                    //    100), 
                    config,
                    true);

                algo.Train(
                    (epocheNumber) =>
                    {
                        return
                            trainData;
                    });
            }

        }
    }
}
