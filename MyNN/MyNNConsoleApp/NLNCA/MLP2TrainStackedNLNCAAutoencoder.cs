using System;
using System.IO;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.BackpropagationFactory;
using MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.GPU;
using MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.GPU;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;

namespace MyNNConsoleApp.NLNCA
{
    public class MLP2TrainStackedNLNCAAutoencoder
    {
        public static void Train()
        {
            var root = "SDAE" + DateTime.Now.ToString("yyyyMMddHHmmss");

            var rndSeed = 7835;
            var randomizer = new DefaultRandomizer(ref rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                1000,
                true);

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                100,
                true);

            var noiser = new SetOfNoisers2(
                randomizer,
                new Pair<float, INoiser>(0.33f, new ZeroMaskingNoiser(randomizer, 0.30f)),
                new Pair<float, INoiser>(0.33f, new SaltAndPepperNoiser(randomizer, 0.30f)),
                new Pair<float, INoiser>(0.34f, new GaussNoiser(0.25f, false))
                );

            var firstLayerSize = trainData[0].Input.Length;

            var serialization = new SerializationHelper();

            const float lambda = 0.1f;
            const float partTakeOfAccount = 0.5f;

            throw new NotImplementedException();

            var sa = new StackedAutoencoder(
                randomizer,
                serialization,
                (DataSet td) =>
                {
                    return 
                        new NoiseDataProvider(
                            td,
                            noiser);
                },
                (DataSet vd) =>
                {
                    return
                        new AutoencoderValidation(
                            new FileSystemMLPSaver(serialization),
                            new RMSE(), 
                            vd.ConvertToAutoencoder(),
                            300,
                            100);
                },
                (int depthIndex) =>
                {
                    var lr = 0.0005f;

                    var conf = new LearningAlgorithmConfig(
                        new LinearLearningRate(lr, 0.99f),
                        100,
                        0.0f,
                        100,
                        0f,
                        -0.001f);

                    return conf;
                },
                //lambda,
                //partTakeOfAccount,
                new GPUBackpropagationAlgorithmFactory(), 
                new GPUForwardPropagationFactory(),
                new LayerInfo(firstLayerSize, new IRLUFunction()),
                new LayerInfo(600, new IRLUFunction()),
                new LayerInfo(600, new IRLUFunction()),
                new LayerInfo(2200, new IRLUFunction())
                );

            if (!Directory.Exists(root))
            {
                Directory.CreateDirectory(root);
            }

            var combinedNet = sa.Train(
                root,
                trainData,
                validationData
                );
        }
    }
}
