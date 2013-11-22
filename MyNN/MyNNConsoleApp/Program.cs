using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using MyNN;
using MyNN.Autoencoders;
using MyNN.BoltzmannMachines.LinearNReLU;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.NeuralNet;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons.Function;
using MyNN.NeuralNet.Train;
using MyNN.NeuralNet.Train.Algo;
using MyNN.NeuralNet.Train.Metrics;
using MyNN.NeuralNet.Train.Validation;
using MyNN.NeuralNet.Train.Validation.NLNCA;

namespace MyNNConsoleApp
{
    class Program
    {
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {
                //pabProfiler.Main2();
                //return;


                var rndSeed = 33514;


                var trainData = MNISTDataProvider.GetDataSet(
                    "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                    1000,
                    true);

                var validationData = MNISTDataProvider.GetDataSet(
                    "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                    100,
                    true);
                validationData = validationData.ConvertToAutoencoder();

                var testData = MNISTDataProvider.GetDataSet(
                    "C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                    100,
                    true);

                var root = ".";

                var network = new MultiLayerNeuralNetwork(
                    root,
                    null,
                    new IFunction[]
                    {
                        null,
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                    },
                    ref rndSeed,
                    new int[]
                    {
                        784,
                        500,
                        100,
                        500,
                        784
                    });

                //var trainer = new TrainNLNCA();
                //trainer.Train(
                //    ref rndSeed,
                //    network,
                //    trainData,
                //    validationData,
                //    new MNISTColorProvider());

                var trainer = new TrainAutoencoderNLNCA();
                trainer.Train(
                    network,
                    trainData,
                    validationData,
                    new MNISTColorProvider());

                //var noiser =
                //    new ZeroMaskingNoiser(ref rndSeed, 0.25f);
                ////new GaussNoiser(0.2f, true);
                ////new SaltAndPepperNoiser(ref rndSeed, 0.25f);

                //var root = "SDAE" + DateTime.Now.ToString("yyyyMMddHHmmss") + "ZMN0.25";

                //var sae = new TrainStackedAutoencoder();
                //sae.Train(
                //    ref rndSeed,
                //    wordDict.Count,
                //    trainData,
                //    validationData,
                //    root,
                //    noiser);

                //var mlp = new TrainMLP();
                //mlp.Train(
                //    trainData,
                //    validationData);


                //var voter = new CreateVote();
                //voter.Create(
                //    testTotalData,
                //    testDataList);

                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
