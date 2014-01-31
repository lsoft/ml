using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict;
using MyNN.MLP2.Backpropagaion.Validation.NLNCA;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2NLNCA
    {

        public static void TrainNLNCA()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                //"_MNIST_DATABASE/mnist/trainingset/",
                //int.MaxValue
                100
                );
            trainData.Normalize();
            //trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                //"_MNIST_DATABASE/mnist/testset/",
                //int.MaxValue
                100
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            int rndSeed = 453123;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var root = ".";
            var folderName = "NLNCAMLP" + DateTime.Now.ToString("yyyyMMddHHmmss") + " MLP2";

            var net = new MLP(
                randomizer,
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
                100,
                0.0f,
                1000,
                0.0001f,
                -1.0f);

            using (var clProvider = new CLProvider())
            {

                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    (processedMLP, processedConfig) => new OpenCLNLNCABackpropagationAlgorithm(
                        VectorizationSizeEnum.VectorizationMode16,
                        processedMLP,
                        processedConfig,
                        clProvider,
                        (uzkii) => new DodfCalculatorOpenCL(
                            uzkii,
                            new VOpenCLDistanceDictFactory())),
                    net,
                    new NLNCAValidation(
                        serialization,
                        trainData,
                        validationData,
                        new MNISTColorProvider(),
                        3), 
                    config);

                algo.Train(
                    new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
            }


        }
    }
}
