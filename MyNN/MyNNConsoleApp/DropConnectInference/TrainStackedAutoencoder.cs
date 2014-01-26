using System;
using System.IO;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Autoencoders.BackpropagationFactory;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNNConsoleApp.DropConnectInference
{
    public class TrainStackedAutoencoder
    {
        public static void Train()
        {

            int rndSeed = 66677890;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //50
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //10
                );
            validationData.Normalize();

            int firstLayerSize = trainData[0].Input.Length;

            var serialization = new SerializationHelper();

            var noiser = new AllNoisers(
                randomizer,
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer)),
                new SaltAndPepperNoiser(randomizer, 0.25f, new RandomRange(randomizer)),
                new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer))
                );

            //var noised = trainData.GetInputPart().Take(300).ToList().ConvertAll(j => noiser.ApplyNoise(j));
            //var v = new MNISTVisualizer();
            //v.SaveAsGrid(
            //    "_allnoisers.bmp",
            //    noised);

            const int sampleCount = 2500;
            const float p = 0.5f;

            var sa = new StackedAutoencoder(
                randomizer,
                serialization,
                (DataSet td) =>
                {
                    return
                        new NoiseDataProvider(
                            td.ConvertToAutoencoder(),
                            noiser);
                },
                (DataSet vd) =>
                {
                    return
                        new AutoencoderValidation(
                            new FileSystemMLPSaver(serialization), 
                            new HalfSquaredEuclidianDistance(), 
                            vd.ConvertToAutoencoder(),
                            400,
                            100);
                },
                (int depthIndex) =>
                {
                    //var lr = 0.005f;

                    //if (depthIndex == 1)
                    //{
                    //    lr = 0.0003f;
                    //}
                    //else if (depthIndex == 2)
                    //{
                    //    lr = 0.00002f;
                    //}

                    var lr = 0.1f;

                    var conf = new LearningAlgorithmConfig(
                        new LinearLearningRate(lr, 0.99f),
                        1,
                        0.0f,
                        100,
                        0f,
                        -0.0025f);

                    return conf;
                },
                new DropConnectBitOpenCLBackpropagationAlgorithmFactory(sampleCount, p),
                new LayerInfo(firstLayerSize, new SigmoidFunction(1f)),
                new LayerInfo(1500, new SigmoidFunction(1f)),
                new LayerInfo(1500, new SigmoidFunction(1f)),
                new LayerInfo(3000, new SigmoidFunction(1f))
                );

            var root = "SDAE" + DateTime.Now.ToString("yyyyMMddHHmmss") + " MLP2";

            if (!Directory.Exists(root))
            {
                Directory.CreateDirectory(root);
            }

            var combinedNet = sa.Train(
                root,
                trainData,
                validationData
                );

            int g = 0;
        }
    }
}
