﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Metadata.W3cXsd2001;
using System.Text;
using MyNN;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
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
using MyNN.MLP2.Structure;
using OpenCL.Net.OpenCL;

namespace MyNNConsoleApp.ClassificationAutoencoder
{
    public class CATuneSCDAE
    {
        public static void Tune()
        {
            var rndSeed = 32323234;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //1000
                );
            trainData.Normalize();
            //trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //100
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            var mlp = SerializationHelper.LoadFromFile<MLP>(
                "SCDAE20140120143731/mlp20140121091839.scdae");
                //"MLP20131218124915/epoche 42/20131219100700-perItemError=3,6219.mynn");
                //"MLP20131219184828/epoche 28/20131220091649-perItemError=2,600619.mynn");

            Console.WriteLine("Network configuration: " + mlp.DumpLayerInformation());


            using (var clProvider = new CLProvider())
            {
                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.001f, 0.99f),
                    1,
                    0.0f,
                    50,
                    0.0001f,
                    -1.0f);

                var validation = new ClassificationAutoencoderValidation(
                    serialization,
                    new HalfSquaredEuclidianDistance(), 
                    validationData,
                    300,
                    100);

                var alg =
                    new BackpropagationAlgorithm(
                        randomizer,
                        (currentMLP, currentConfig) =>
                            new OpenCLBackpropagationAlgorithm(
                                VectorizationSizeEnum.VectorizationMode16,
                                currentMLP,
                                currentConfig,
                                clProvider),
                        mlp,
                        validation,
                        config);

                var noiser = new AllNoisers(
                    randomizer,
                    new ZeroMaskingNoiser(randomizer, 0.20f, new RandomRange(randomizer)),
                    new SaltAndPepperNoiser(randomizer, 0.20f, new RandomRange(randomizer)),
                    new GaussNoiser(0.175f, false, new RandomRange(randomizer)),
                    new MultiplierNoiser(randomizer, 0.75f, new RandomRange(randomizer)),
                    new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer))
                    );


                //обучение сети
                alg.Train(
                    //new NoDeformationTrainDataProvider(trainData.ConvertToAutoencoder()).GetDeformationDataSet);
                    new NoiseDataProvider(trainData.ConvertToClassificationAutoencoder(), noiser).GetDeformationDataSet);
            }


        }
    }
}