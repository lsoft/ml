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
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.BackpropagationFactory;
using MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.CPU;
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

namespace MyNNConsoleApp.ClassificationAutoencoder
{
    public class CATrainSCDAE
    {
        public static void Train()
        {

            int rndSeed = 1888227;
            var randomizer = new DefaultRandomizer(++rndSeed);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                //100
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                //100
                );
            validationData.Normalize();

            int firstLayerSize = trainData[0].Input.Length;

            var serialization = new SerializationHelper();

            //var noiser = new SetOfNoisers(
            //    randomizer,
            //    new Pair<float, INoiser>(0.33f, new ZeroMaskingNoiser(randomizer, 0.25f)),
            //    new Pair<float, INoiser>(0.33f, new SaltAndPepperNoiser(randomizer, 0.25f)),
            //    new Pair<float, INoiser>(0.34f, new GaussNoiser(0.20f, false))
            //    );

            var depth0Noiser = new AllNoisers(
                randomizer,
                new ZeroMaskingNoiser(randomizer, 0.15f, new RandomRange(randomizer)),
                new SaltAndPepperNoiser(randomizer, 0.15f, new RandomRange(randomizer)),
                new GaussNoiser(0.15f, false, new RandomRange(randomizer)),
                new MultiplierNoiser(randomizer, 0.5f, new RandomRange(randomizer)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer))
                );

            //var depthNot0Noiser = new AllNoisers(
            //    randomizer,
            //    new SaltAndPepperNoiser(randomizer, 0.15f, new RandomRange(randomizer)),
            //    new GaussNoiser(0.5f, false, new RandomRange(randomizer)),
            //    new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer))
            //    );

            //var noised = trainData.GetInputPart().Take(300).ToList().ConvertAll(j => noiser.ApplyNoise(j));
            //var v = new MNISTVisualizer();
            //v.SaveAsGrid(
            //    "_allnoisers.bmp",
            //    noised);

            var layerFactory = new LayerFactory(new NeuronFactory(randomizer));

            var scdae = new StackedClassificationAutoencoder(
                new IntelCPUDeviceChooser(), 
                randomizer,
                new MLPFactory(
                    layerFactory), 
                serialization,
                (int depthIndex, DataSet td) =>
                {
                    //if (depthIndex == 0)
                    //{
                        return
                            new NoiseDataProvider(
                                td.ConvertToClassificationAutoencoder(),
                                depth0Noiser);
                    //}
                    //else
                    //{
                    //    return
                    //        new NoiseDataProvider(
                    //            td.ConvertToClassificationAutoencoder(),
                    //            depthNot0Noiser);
                    //            //depth0Noiser);
                    //}
                },
                (DataSet vd) =>
                {
                    return
                        new ClassificationAutoencoderValidation(
                            new FileSystemMLPSaver(serialization), 
                            new HalfSquaredEuclidianDistance(),
                            vd,
                            300,
                            100);
                },
                (int depthIndex) =>
                {
                    var lr =
                        depthIndex == 0
                            ? 0.005f
                            : 0.001f;
                        //0.1f;

                    var conf = new LearningAlgorithmConfig(
                        new LinearLearningRate(lr, 0.99f),
                        1,
                        0.0f,
                        100,
                        0f,
                        -0.0025f);

                    return conf;
                },
                //new GPUBackpropagationAlgorithmFactory(),
                new CPUBackpropagationAlgorithmFactory(),
                //new GPUForwardPropagationFactory(),
                new CPUForwardPropagationFactory(),
                //new LayerInfo(firstLayerSize, new RLUFunction()),
                //new LayerInfo(800, new RLUFunction()),
                //new LayerInfo(800, new RLUFunction()),
                //new LayerInfo(1600, new RLUFunction())
                new LayerInfo(firstLayerSize, new SigmoidFunction(1f)),
                new LayerInfo(1200, new SigmoidFunction(1f)),
                new LayerInfo(1200, new SigmoidFunction(1f)),
                new LayerInfo(2200, new RLUFunction())
                );

            var root = "SCDAE" + DateTime.Now.ToString("yyyyMMddHHmmss");

            if (!Directory.Exists(root))
            {
                Directory.CreateDirectory(root);
            }

            var combinedNet = scdae.Train(
                root,
                trainData,
                validationData
                );

            int g = 0;
        }
    }
}
