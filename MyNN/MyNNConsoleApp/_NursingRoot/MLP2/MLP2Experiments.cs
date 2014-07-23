using System;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Container;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2Experiments
    {
        public static void Train()
        {
            int rndSeed = 123;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var layerFactory = new LayerFactory(new NeuronFactory(randomizer));
            

            var mlpf = new MLPFactory(
                layerFactory
                );

            var mlp = mlpf.CreateMLP(
                DateTime.Now.Ticks.ToString(),
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
            //    var forward = new CPUForwardPropagation(
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
                    new CPUBackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider),
                    new FileSystemMLPContainer(".", serialization),
                    mlp,
                    new ClassificationValidation(
                        
                        new HalfSquaredEuclidianDistance(), 
                        validationData,
                        300,
                        100), 
                    //new AutoencoderValidation(
                    //    validationData,
                    //    100,
                    //    100), 
                    config);

                algo.Train(new NoDeformationTrainDataProvider(trainData));
            }

        }
    }
}
