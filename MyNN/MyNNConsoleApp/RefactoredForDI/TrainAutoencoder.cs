﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer;
using MyNN.MLP2.Container;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainAutoencoder
    {
        public static void DoTrain()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000 //int.MaxValue
                );
            trainData.Normalize();
            trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300 //int.MaxValue
                );
            validationData.Normalize();
            validationData = validationData.ConvertToAutoencoder();

            var randomizer = new DefaultRandomizer(123);

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemMLPContainer(
                ".",
                serialization);

            var validation = new Validation(
                new MetricsAccuracyCalculator( 
                    new HalfSquaredEuclidianDistance(),
                    validationData),
                new GridReconstructDrawer(
                    new MNISTVisualizer(), 
                    validationData,
                    300,
                    100)
                );

            var config = new LearningAlgorithmConfig(
                new LinearLearningRate(0.1f, 0.99f),
                1,
                0f,
                15,
                -1f,
                -1f
                );

            var trainDataProvider = new NoDeformationTrainDataProvider(trainData);

            var autoencoderName = string.Format(
                "ae{0}.ae",
                DateTime.Now.ToString("yyyyMMddHHmmss"));

            var mlp = mlpfactory.CreateMLP(
                autoencoderName,
                new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f), 
                        new SigmoidFunction(1f), 
                        new SigmoidFunction(1f), 
                        new SigmoidFunction(1f), 
                    },
                new int[]
                    {
                        784,
                        500,
                        100,
                        500,
                        784
                    });

            var mlpContainer = rootContainer.GetChildContainer(autoencoderName);

            using (var clProvider = new CLProvider())
            {
                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    new CPUBackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider),
                    mlpContainer,
                    mlp,
                    validation,
                    config);

                algo.Train(trainDataProvider);
            }
        }
    }
}