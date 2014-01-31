﻿using System;
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
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.CPU.Default;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.CPU.Transpose;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.CPU.Transpose2;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.MLP2.Transposer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.TransposeExperiments
{

    public class Transpose2
    {
        public static void Execute()
        {
            const int batchSize = 25;

            const int trainDataCount = 250;
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                trainDataCount
                );
            trainData.Normalize();
            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                //int.MaxValue
                25
                );
            validationData.Normalize();
            var layerSizes = new[] { 784, 331, 779, 784 };

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
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                    },
                    layerSizes);


                var conf = new LearningAlgorithmConfig(
                    new ConstLearningRate(0.001f),
                    batchSize,
                    0.0f,
                    1,
                    0f,
                    -0.0025f);

                var validation = new AutoencoderValidation(
                    new FileSystemMLPSaver(serialization),
                    new HalfSquaredEuclidianDistance(),
                    validationData.ConvertToAutoencoder(),
                    300,
                    100);

                using (var clProvider = new CLProvider())
                {
                    var algo = new BackpropagationAlgorithm(
                        randomizer,
                        (processedMLP, processedConfig) => new CPUBackpropagationAlgorithm(
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
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                    },
                    layerSizes);


                var conf = new LearningAlgorithmConfig(
                    new ConstLearningRate(0.001f),
                    batchSize,
                    0.0f,
                    1,
                    0f,
                    -0.0025f);

                var validation = new AutoencoderValidation(
                    new FileSystemMLPSaver(serialization),
                    new HalfSquaredEuclidianDistance(),
                    validationData.ConvertToAutoencoder(),
                    300,
                    100);

                using (var clProvider = new CLProvider())
                {
                    var algo = new BackpropagationAlgorithm(
                        randomizer,
                        (processedMLP, processedConfig) => new CPUTransposeBackpropagationAlgorithm(
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

            MLP mlp3 = null;
            {
                Console.WriteLine("============================== NEW 2 ==================================");

                var randomizer = new NoRandomRandomizer();

                var folderName = "_TransposeAutoencoder" + DateTime.Now.ToString("yyyyMMddHHmmss") + " MLP2";

                mlp3 = new MLP(
                    randomizer,
                    ".",
                    folderName,
                    new IFunction[]
                    {
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                    },
                    layerSizes);


                var conf = new LearningAlgorithmConfig(
                    new ConstLearningRate(0.001f),
                    batchSize,
                    0.0f,
                    1,
                    0f,
                    -0.0025f);

                var validation = new AutoencoderValidation(
                    new FileSystemMLPSaver(serialization),
                    new HalfSquaredEuclidianDistance(),
                    validationData.ConvertToAutoencoder(),
                    300,
                    100);

                using (var clProvider = new CLProvider())
                {
                    var algo = new BackpropagationAlgorithm(
                        randomizer,
                        (processedMLP, processedConfig) => new CPUTranspose2BackpropagationAlgorithm(
                            VectorizationSizeEnum.VectorizationMode16,
                            processedMLP,
                            processedConfig,
                            clProvider),
                        mlp3,
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
                        var diff = Math.Abs(mlp1.Layers[ll].Neurons[nn].Weights[ww] - mlp3.Layers[ll].Neurons[nn].Weights[ww]);
                        if (maxdiff < diff)
                        {
                            maxdiff = diff;
                        }
                    }
                }
            }

            Console.WriteLine("EXPERIMENT: Transpose2");
            Console.WriteLine("Max diff = {0}", maxdiff);
            Console.ReadLine();
        }
    }
}
