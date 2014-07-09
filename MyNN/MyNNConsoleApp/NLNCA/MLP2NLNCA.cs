using System;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.KNN;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.ClassificationMLP.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation1;
using MyNN.MLP2.Backpropagation.Validation.NLNCA;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.NLNCA
{
    public class MLP2NLNCA
    {

        public static void TrainNLNCA()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                //int.MaxValue
                500
                );
            trainData.Normalize();
            //trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                //int.MaxValue
                100
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            int rndSeed = 453124;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var root = ".";
            var folderName = "NLNCAMLP" + DateTime.Now.ToString("yyyyMMddHHmmss") + " MLP2";

            var layerFactory = new LayerFactory(new NeuronFactory(randomizer));
            

            var mlpf = new MLPFactory(
                layerFactory
                );

            var mlp = mlpf.CreateMLP(
                root,
                folderName,
                new IFunction[3]
                    {
                        null,
                        new SigmoidFunction(1f), 
                        new LinearFunction(1f)
                    },
                new int[3]
                    {
                        784,
                        300,
                        2
                    });

            var config = new LearningAlgorithmConfig(
                new LinearLearningRate(0.5f, 0.99f),
                500,
                0.0f,
                1000,
                0.0001f,
                -1.0f);

            using (var clProvider = new CLProvider())
            {

                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    new CPUNLNCABackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider,
                        (uzkii) => new DodfCalculatorOpenCL(
                            uzkii,
                            new VectorizedCpuDistanceDictCalculator())),
                    mlp,
                    new NLNCAValidation(
                        new CPUOpenCLKNearestFactory(), 
                        serialization,
                        trainData,
                        validationData,
                        new MNISTColorProvider(),
                        3), 
                    config);

                algo.Train(
                    new NoDeformationTrainDataProvider(trainData));
            }


        }
    }
}
