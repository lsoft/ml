﻿using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNNConsoleApp.PingPong
{
    public class NextAutoencoder
    {
        public static void Execute()
        {
            var rndSeed = 665341;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //100
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //100
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            //через обученную сеть генерируем данные для следующей эпохи
            var mlpPath = "PingPong/Experiment0/Step0/20140115132416-9876 out of 98,76%.mynn";

            DataSet trainNext;
            DataSet validationNext;
            NextDataSet.NextDataSets(mlpPath, trainData, validationData, out trainNext, out validationNext);

            //обучаем автоенкодер
            var a = new Autoencoder(
                randomizer,
                "PingPong/Experiment0",
                null,
                new LayerInfo[]
                {
                    new LayerInfo(
                        trainNext[0].Input.Length,
                        new RLUFunction()), 
                    new LayerInfo(
                        1200,
                        new RLUFunction()),
                    new LayerInfo(
                        trainNext[0].Input.Length,
                        new RLUFunction()), 
                });

            var config = new LearningAlgorithmConfig(
                new LinearLearningRate(0.001f, 0.99f),
                1,
                0.0f,
                50,
                0.0001f,
                -1.0f);

            var noiser = new SetOfNoisers(
                randomizer,
                new Pair<float, INoiser>(0.33f, new ZeroMaskingNoiser(randomizer, 0.25f)),
                new Pair<float, INoiser>(0.33f, new SaltAndPepperNoiser(randomizer, 0.25f)),
                new Pair<float, INoiser>(0.34f, new GaussNoiser(0.20f, false))
                );

            var validation = new AutoencoderValidation(
                serialization,
                new HalfSquaredEuclidianDistance(),
                validationNext.ConvertToAutoencoder(),
                300,
                100);

            a.Train(
                config,
                new NoiseDataProvider(trainNext.ConvertToAutoencoder(), noiser),
                validation);
        }
    }
}
