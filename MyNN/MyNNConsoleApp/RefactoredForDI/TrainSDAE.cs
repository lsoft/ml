using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Autoencoders;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer;
using MyNN.MLP2.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainSDAE
    {
        public static void DoTrain()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000 //int.MaxValue
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300 //int.MaxValue
                );
            validationData.Normalize();

            var randomizer = new DefaultRandomizer(123);

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var noiser = new AllNoisers(
                randomizer,
                new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomRange(randomizer)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer))
                );

            var sdae = new StackedAutoencoder(
                new IntelCPUDeviceChooser(),
                randomizer,
                mlpfactory,
                (IDataSet td) =>
                {
                    return
                        new NoiseDataProvider(
                            td.ConvertToAutoencoder(),
                            noiser);
                },
                (IDataSet vd, IArtifactContainer mlpContainer) =>
                {
                    var vda = vd.ConvertToAutoencoder();

                    return
                        new Validation(
                            new MetricsAccuracyCalculator(
                                new HalfSquaredEuclidianDistance(),
                                vda),
                            new GridReconstructDrawer(
                                new MNISTVisualizer(), 
                                vda,
                                300,
                                100)
                            );
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
                        15,
                        0f,
                        -0.0025f);

                    return conf;
                },
                new CPUBackpropagationAlgorithmFactory(),
                new CPUForwardPropagationFactory(),
                new LayerInfo(784, new RLUFunction()),
                new LayerInfo(500, new RLUFunction()),
                new LayerInfo(500, new RLUFunction()),
                new LayerInfo(1000, new RLUFunction())
                );

            var sdaeName = string.Format(
                "sdae{0}.sdae",
                DateTime.Now.ToString("yyyyMMddHHmmss"));

            sdae.Train(
                sdaeName,
                rootContainer,
                trainData,
                validationData
                );
        }
    }
}
