using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCLTranspose;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.MLP2.Transposer;
using OpenCL.Net.OpenCL;

namespace MyNNConsoleApp.TransposeExperiments
{

    public class Transpose1
    {
        public static void Execute()
        {
            const int batchSize = 1;

            //const int trainDataCount = 1;
            //var trainData = MNISTDataProvider.GetDataSet(
            //    "_MNIST_DATABASE/mnist/trainingset/",
            //    trainDataCount
            //    );
            //trainData.Normalize();
            //var validationData = MNISTDataProvider.GetDataSet(
            //    "_MNIST_DATABASE/mnist/testset/",
            //    //int.MaxValue
            //    100
            //    );
            //validationData.Normalize();
            //var layerSizes = new[] { 784, 12, 784 };

            const int firstItemLength = 5;
            var trainData = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new float[firstItemLength]
                        {
                            1f,
                            2f,
                            3f,
                            4f,
                            5f
                        },
                        new float[]
                        {
                            1f
                        })
                });

            var validationData = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new float[firstItemLength]
                        {
                            1f,
                            2f,
                            3f,
                            4f,
                            5f
                        },
                        new float[]
                        {
                            1f
                        })
                });

            var layerSizes = new[] { firstItemLength, 2, firstItemLength };

            var serialization = new SerializationHelper();

            MLP mlp1 = null;
            {
                Console.WriteLine("============================== OLD ==================================");

                var randomizer = new NoRandomRandomizer();

                var folderName = "_TransposeAutoencoder" + DateTime.Now.ToString("yyyyMMddHHmmss") + " MLP2";

                mlp1 = new MLP(
                    randomizer,
                    ".",
                    folderName,
                    new IFunction[]
                    {
                        new LinearFunction(1f), 
                        new LinearFunction(1f), 
                        new LinearFunction(1f)
                    },
                    layerSizes);


                var conf = new LearningAlgorithmConfig(
                    new ConstLearningRate(1f),
                    batchSize,
                    0.0f,
                    1,
                    0f,
                    -0.0025f);

                var validation = new AutoencoderValidation(
                    serialization,
                    new HalfSquaredEuclidianDistance(),
                    validationData.ConvertToAutoencoder(),
                    300,
                    100);

                using (var clProvider = new CLProvider())
                {
                    var algo = new BackpropagationAlgorithm(
                        randomizer,
                        (processedMLP, processedConfig) => new OpenCLBackpropagationAlgorithm(
                            VectorizationSizeEnum.VectorizationMode16,
                            processedMLP,
                            processedConfig,
                            clProvider),
                        mlp1,
                        validation,
                        conf,
                        true);

                    algo.Train(
                        new NoDeformationTrainDataProvider(
                            trainData.ConvertToAutoencoder()).GetDeformationDataSet);
                }
            }

            MLP mlp2 = null;
            {
                Console.WriteLine("============================== NEW ==================================");

                var randomizer = new NoRandomRandomizer();

                var folderName = "_TransposeAutoencoder" + DateTime.Now.ToString("yyyyMMddHHmmss") + " MLP2";

                mlp2 = new MLP(
                    randomizer,
                    ".",
                    folderName,
                    new IFunction[]
                    {
                        new LinearFunction(1f), 
                        new LinearFunction(1f), 
                        new LinearFunction(1f)
                    },
                    layerSizes);


                var conf = new LearningAlgorithmConfig(
                    new ConstLearningRate(1f),
                    batchSize,
                    0.0f,
                    1,
                    0f,
                    -0.0025f);

                var validation = new AutoencoderValidation(
                    serialization,
                    new HalfSquaredEuclidianDistance(),
                    validationData.ConvertToAutoencoder(),
                    300,
                    100);

                using (var clProvider = new CLProvider())
                {
                    var algo = new BackpropagationAlgorithm(
                        randomizer,
                        (processedMLP, processedConfig) => new OpenCLTransposeBackpropagationAlgorithm(
                            VectorizationSizeEnum.VectorizationMode16,
                            processedMLP,
                            processedConfig,
                            clProvider),
                        mlp2,
                        validation,
                        conf,
                        true);

                    algo.Train(
                        new NoDeformationTrainDataProvider(
                            trainData.ConvertToAutoencoder()).GetDeformationDataSet);
                }
                //*/
            }

            //высчитываем ошибку
            float maxdiff = 0f;

            for (var ll = 0; ll < mlp1.Layers.Length; ll++)
            {
                for (var nn = 0; nn < mlp1.Layers[ll].Neurons.Length; nn++)
                {
                    for (var ww = 0; ww < mlp1.Layers[ll].Neurons[nn].Weights.Length; ww++)
                    {
                        var diff = Math.Abs(mlp1.Layers[ll].Neurons[nn].Weights[ww] - mlp2.Layers[ll].Neurons[nn].Weights[ww]);
                        if (maxdiff < diff)
                        {
                            maxdiff = diff;
                        }
                    }
                }
            }

            Console.WriteLine("EXPERIMENT: Transpose1");
            Console.WriteLine("Max diff = {0}", maxdiff);
            Console.ReadLine();
        }
    }
}
