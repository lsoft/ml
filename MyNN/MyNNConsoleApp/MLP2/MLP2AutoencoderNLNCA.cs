using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.AutoencoderMLP.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation1;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
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
    public class MLP2AutoencoderNLNCA
    {

        public static void TrainAutoencoderNLNCA()
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
                1000
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            int rndSeed = 453123;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var root = ".";
            var folderName = "NLNCA Autoencoder" + DateTime.Now.ToString("yyyyMMddHHmmss") + " MLP2";

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
                100,
                0.0f,
                50,
                0.0001f,
                -1.0f);

            using (var clProvider = new CLProvider())
            {

                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    new CPUAutoencoderNLNCABackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider,
                        (uzkii) => new DodfCalculatorOpenCL(
                            uzkii,
                            new VectorizedCpuDistanceDictCalculator()),
                        1,
                        0.9f,
                        50),
                    mlp,
                    //new NLNCAValidation(
                    //    //new RMSE(),
                    //    trainData,
                    //    validationData,
                    //    new MNISTColorProvider(),
                    //    3), 
                    new AutoencoderValidation(
                        new FileSystemMLPSaver(serialization),
                        new RMSE(),
                        validationData.ConvertToAutoencoder(),
                        300,
                        100), 
                    config);

                algo.Train(
                    new NoDeformationTrainDataProvider(trainData));
            }
            //*/

        }
    }
}
