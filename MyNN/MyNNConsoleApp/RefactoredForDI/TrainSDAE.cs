using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Data.TrainDataProvider.Noiser.Range;
using MyNN.Common.Data.TypicalDataProvider;
using MyNN.Common.LearningRateController;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Autoencoders;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.ForwardPropagationFactory;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
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
                int.MaxValue
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                );
            validationData.Normalize();

            var randomizer = new DefaultRandomizer(123);

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var toa = new ToAutoencoderDataSetConverter();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var noiser = new SequenceNoiser(
                randomizer,
                true,
                new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer, trainData[0].InputLength))
                );

            var mlpContainerHelper = new MLPContainerHelper();

            using (var clProvider = new CLProvider())
            {
                var sdae = new StackedAutoencoder(
                    new IntelCPUDeviceChooser(),
                    mlpContainerHelper,
                    randomizer,
                    mlpfactory,
                    (int depthIndex, IDataSet td) =>
                    {
                        var tda = toa.Convert(td);

                        var result =
                            new ConverterTrainDataProvider(
                                new ShuffleDataSetConverter(randomizer),
                                new NoiseDataProvider(tda, noiser)
                                );
                        return
                            result;
                    },
                    (int depthIndex, IDataSet vd, IArtifactContainer mlpContainer) =>
                    {
                        var vda = toa.Convert(vd);

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
                            50,
                            0f,
                            -0.0025f);

                        return conf;
                    },
                    new CPUBackpropagationFactory(
                        mlpContainerHelper),
                    new ForwardPropagationFactory(
                        new CPUPropagatorComponentConstructor(
                            clProvider,
                            VectorizationSizeEnum.VectorizationMode16)),
                    new LayerInfo(784, new RLUFunction()),
                    new LayerInfo(1000, new RLUFunction()),
                    new LayerInfo(1000, new RLUFunction()),
                    new LayerInfo(2200, new RLUFunction())
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
}
