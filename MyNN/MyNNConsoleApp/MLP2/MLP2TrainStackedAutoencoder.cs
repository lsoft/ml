using System;
using System.IO;
using System.Linq;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.BackpropagationFactory;
using MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;

namespace MyNNConsoleApp.MLP2
{
    public class MLP2TrainStackedAutoencoder
    {
        public static void Train()
        {

            int rndSeed = 399317;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //500
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
            //validationData = validationData.ConvertToAutoencoder();

            int firstLayerSize = trainData[0].Input.Length;

            var serialization = new SerializationHelper();

            //var noiser = new SetOfNoisers(
            //    randomizer,
            //    new Pair<float, INoiser>(0.33f, new ZeroMaskingNoiser(randomizer, 0.25f)),
            //    new Pair<float, INoiser>(0.33f, new SaltAndPepperNoiser(randomizer, 0.25f)),
            //    new Pair<float, INoiser>(0.34f, new GaussNoiser(0.20f, false))
            //    );

            var noiser = new AllNoisers(
                randomizer,
                new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomRange(randomizer)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer))
                );

            //var noised = trainData.GetInputPart().Take(300).ToList().ConvertAll(j => noiser.ApplyNoise(j));
            //var v = new MNISTVisualizer();
            //v.SaveAsGrid(
            //    "_allnoisers.bmp",
            //    noised);

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
                            300,
                            100);
                },
                (int depthIndex) =>
                {
                    var lr =
                        depthIndex == 0
                            ? 0.005f
                            : 0.001f;

                    var conf = new LearningAlgorithmConfig(
                        new LinearLearningRate(lr, 0.99f),
                        1,
                        0.0f,
                        50,
                        0f,
                        -0.0025f);

                    return conf;
                },
                new CPUBackpropagationAlgorithmFactory(),
                new CPUForwardPropagationFactory(),
                new LayerInfo(firstLayerSize, new RLUFunction()),
                new LayerInfo(1200, new RLUFunction()),
                new LayerInfo(1200, new RLUFunction()),
                new LayerInfo(2200, new RLUFunction())
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
