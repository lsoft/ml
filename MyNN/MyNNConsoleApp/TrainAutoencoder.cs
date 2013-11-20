using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Autoencoders;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons.Function;
using MyNN.NeuralNet.Train.Metrics;
using MyNN.NeuralNet.Train.Validation;

namespace MyNNConsoleApp
{
    public class TrainAutoencoder
    {
        public void Train(
            ref int rndSeed,
            INoiser noiser)
        {
            if (noiser == null)
            {
                throw new ArgumentNullException("noiser");
            }

            var trainData = MNISTDataProvider.GetDataSet(
                "mnist/trainingset/",
                int.MaxValue);
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "mnist/testset/",
                int.MaxValue);
            validationData.Normalize();

            var a = new Autoencoder(
                ref rndSeed,
                null,
                null,
                new LayerInfo(784, new SigmoidFunction(1f)),
                new LayerInfo(2000, new RLUFunction()),
                new LayerInfo(800, new RLUFunction()),
                new LayerInfo(2000, new RLUFunction()),
                new LayerInfo(784, new SigmoidFunction(1f))
                );

            var conf = new LearningAlgorithmConfig(
                new LinearLearningRate(0.01f, 0.99f),
                1,
                0.0f,
                1000,
                0.0001f,
                -1.0f,
                new HalfSquaredEuclidianDistance());

            a.Train(
                ref rndSeed,
                conf,
                //new NoDeformationTrainDataProvider(trainData.ConvertToAutoencoder()),
                new NoiseDataProvider(
                    trainData.ConvertToAutoencoder(),
                    noiser),
                new AutoencoderValidation(validationData.ConvertToAutoencoder(), 300, 100)
                );
        }
    }
}
