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
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.Backpropagaion.Validation.NLNCA;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.OpenCL;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2AutoencoderNLNCA
    {

        public static void TrainAutoencoderNLNCA()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                //"_MNIST_DATABASE/mnist/trainingset/",
                //int.MaxValue
                1000
                );
            trainData.Normalize();
            //trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                //"_MNIST_DATABASE/mnist/testset/",
                //int.MaxValue
                1000
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            int rndSeed = 453123;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var root = ".";
            var folderName = "NLNCA Autoencoder" + DateTime.Now.ToString("yyyyMMddHHmmss") + " MLP2";

            var net = new MLP(
                randomizer,
                root,
                folderName,
                new IFunction[3]
                    {
                        null,
                        new LinearFunction(1f),
                        new SigmoidFunction(1f)
                    },
                new int[3]
                    {
                        784,
                        100,
                        784
                    });

            var config = new LearningAlgorithmConfig(
                new LinearLearningRate(0.2f, 0.99f),
                25,
                0.0f,
                50,
                0.0001f,
                -1.0f);

            /*
            using (var clProvider = new CLProvider())
            {

                var algo = new BackpropagationAlgorithm(
                    (processedMLP, processedConfig) => new DropConnectOpenCLBackpropagationAlgorithm(
                        OpenCLForwardPropagation.VectorizationSizeEnum.VectorizationMode16,
                        processedMLP,
                        processedConfig,
                        clProvider),
                    net,
                    new AutoencoderValidation(
                        new RMSE(),
                        validationData.ConvertToAutoencoder(),
                        300,
                        100),
                    config);

                algo.Train(
                    new NoDeformationTrainDataProvider(trainData.ConvertToAutoencoder()).GetDeformationDataSet);
            }
            //*/

            
            using (var clProvider = new CLProvider())
            {

                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    (processedMLP, processedConfig) => new OpenCLAutoencoderNLNCABackpropagationAlgorithm(
                        VectorizationSizeEnum.VectorizationMode16,
                        processedMLP,
                        processedConfig,
                        clProvider,
                        (uzkii) => new DodfCalculatorOpenCL(
                            uzkii,
                            new VOpenCLDistanceDictFactory()),
                        1,
                        0.9f,
                        50),
                    net,
                    //new NLNCAValidation(
                    //    //new RMSE(),
                    //    trainData,
                    //    validationData,
                    //    new MNISTColorProvider(),
                    //    3), 
                    new AutoencoderValidation(
                        serialization,
                        new RMSE(),
                        validationData.ConvertToAutoencoder(),
                        300,
                        100), 
                    config);

                algo.Train(
                    new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
            }
            //*/

        }
    }
}
