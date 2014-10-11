using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item.Dense;
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
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.ForwardPropagationFactory;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation1;
using MyNN.MLP.NLNCA.BackpropagationFactory.OpenCL.CPU;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainNLNCAAutoencoder
    {
        public static void DoTrain()
        {
            var dataItemFactory = new DenseDataItemFactory();

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000,
                true,
                dataItemFactory
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300,
                true,
                dataItemFactory
                );
            validationData.Normalize();

            var randomizer = new DefaultRandomizer(123);

            var toa = new ToAutoencoderDataSetConverter(
                dataItemFactory
                );

            var noiser = new SequenceNoiser(
                randomizer,
                true,
                new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer, trainData[0].InputLength))
                );

            var serialization = new SerializationHelper();

            var mlpf = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var firstLayerSize = trainData[0].Input.Length;
            const float lambda = 0.5f;
            const float partTakeOfAccount = 0.5f;

            var mlpContainerHelper = new MLPContainerHelper();
            
            var sa = new StackedAutoencoder(
                new IntelCPUDeviceChooser(),
                mlpContainerHelper,
                randomizer,
                dataItemFactory,
                mlpf,
                (int depthIndex, IDataSet td) =>
                {
                    var tda = toa.Convert(td);

                    var result =
                        new ConverterTrainDataProvider(
                            new ShuffleDataSetConverter(randomizer),
                            new NoiseDataProvider(tda, noiser, dataItemFactory)
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
                                new RMSE(),
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
                        250,
                        0.0f,
                        25,
                        0f,
                        -0.0025f);

                    return conf;
                },
                new CPUNLNCABackpropagationFactory(
                    mlpContainerHelper,
                    (data) =>
                        new DodfCalculatorOpenCL(
                            data,
                            new VectorizedCpuDistanceDictCalculator() //generation 1
                            ),
                    1,
                    lambda,
                    partTakeOfAccount),
                (clProvider) => new ForwardPropagationFactory(
                    new CPUPropagatorComponentConstructor(
                        clProvider,
                        VectorizationSizeEnum.VectorizationMode16)),
                new LayerInfo(firstLayerSize, new RLUFunction()),
                new LayerInfo(400, new RLUFunction()),
                new LayerInfo(800, new RLUFunction())
                );

            var mlpName = string.Format(
                "nlnca_ae{0}.sdae",
                DateTime.Now.ToString("yyyyMMddHHmmss"));


            var combinedNet = sa.Train(
                mlpName,
                new FileSystemArtifactContainer(".", serialization),
                trainData,
                validationData
                );
        }
    }
}
