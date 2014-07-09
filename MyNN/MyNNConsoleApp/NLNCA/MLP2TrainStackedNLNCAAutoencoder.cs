using System;
using System.Collections.Generic;
using System.IO;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation3;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation3.Float;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.BackpropagationFactory;
using MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.GPU;
using MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.GPU;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.Randomizer;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.NLNCA
{
    public class MLP2TrainStackedNLNCAAutoencoder
    {
        public static void Train()
        {
            var root = "SDAE" + DateTime.Now.ToString("yyyyMMddHHmmss") + " NLNCA";

            var rndSeed = 7836;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/trainingset/",
                "_MNIST_DATABASE/mnist/trainingset/",
                1000,
                true);

            var validationData = MNISTDataProvider.GetDataSet(
                //"C:/projects/ml/MNIST/_MNIST_DATABASE/mnist/testset/",
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue,
                true);

            var noiser = new AllNoisers(
                randomizer,
                new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomRange(randomizer)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer))
                );

            var firstLayerSize = trainData[0].Input.Length;

            var serialization = new SerializationHelper();

            const float lambda = 0.1f;
            const float partTakeOfAccount = 0.5f;

            var layerFactory = new LayerFactory(new NeuronFactory(randomizer));
            

            var mlpf = new MLPFactory(
                layerFactory
                );

            var sa = new StackedAutoencoder(
                new IntelCPUDeviceChooser(), 
                randomizer,
                mlpf,
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
                    var lr =
                        depthIndex == 0
                            ? 0.005f
                            : 0.001f;

                    var conf = new LearningAlgorithmConfig(
                        new LinearLearningRate(lr, 0.99f),
                        500,
                        0.0f,
                        50,
                        0f,
                        -0.0025f);

                    return conf;
                },
                new CPUNLNCABackpropagationAlgorithmFactory(
                    (data) => 
                        new DodfCalculatorOpenCL(
                            data,
                            new VectorizedCpuDistanceDictCalculator() //generation 3
                            ),
                    1,
                    lambda,
                    partTakeOfAccount), 
                new CPUForwardPropagationFactory(),
                new LayerInfo(firstLayerSize, new RLUFunction()),
                new LayerInfo(600, new RLUFunction()),
                new LayerInfo(600, new RLUFunction()),
                new LayerInfo(2200, new RLUFunction())
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
