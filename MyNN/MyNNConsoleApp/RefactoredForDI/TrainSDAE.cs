using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.DataLoader;

using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Data.TrainDataProvider.Noiser.Range;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Autoencoders;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP.Classic.BackpropagationFactory.Classic.OpenCL.GPU;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU;
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
            var randomizer = new DefaultRandomizer(123);

            var dataItemFactory = new DataItemFactory();

            const int trainMaxCountFilesInCategory = 1000;
            const int validationMaxCountFilesInCategory = 300;

            var trainData = GetTrain(
                trainMaxCountFilesInCategory,
                false,
                true
                );

            var validationData = GetValidation(
                validationMaxCountFilesInCategory,
                false,
                true
                );

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var mlpContainerHelper = new MLPContainerHelper();


            var iteratorFactory = new DataIteratorFactory();

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    //это используется для преобразования данных для следующего автоенкодера
                    //внутри стекед автоенкодера
                    //соотв. здесь никакого преобразования не требуется

                    return 
                        new NoConvertDataItemTransformation();
                });


            var dataSetFactory = new DataSetFactory(
                iteratorFactory,
                itemTransformationFactory
                );

            var sdae = new StackedAutoencoder(
                new NvidiaOrAmdGPUDeviceChooser(true), 
                mlpContainerHelper,
                randomizer,
                dataSetFactory,
                dataItemFactory,
                mlpfactory,
                (int depthIndex, IDataSet td) =>
                {
                    var noiser = new SequenceNoiser(
                        randomizer,
                        true,
                        new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer)),
                        new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer)),
                        new DistanceChangeNoiser(randomizer, 1f, 3, new RandomSeriesRange(randomizer)),
                        new SaltAndPepperNoiser(randomizer, 0.1f, new RandomSeriesRange(randomizer)),
                        new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer))
                        );

                    var iteratorFactory2 = new DataIteratorFactory();

                    var itemTransformationFactory2 = new DataItemTransformationFactory(
                        (epochNumber) =>
                        {
                            return 
                                new NoiserDataItemTransformation(
                                    dataItemFactory,
                                    epochNumber,
                                    noiser,
                                    null
                                    );
                        });


                    var dataSetFactory2 = new DataSetFactory(
                        iteratorFactory2,
                        itemTransformationFactory2
                        );

                    var result = new DataSetProvider(
                        dataSetFactory2,
                        new FromArrayDataItemLoader(
                            td,
                            new DefaultNormalizer())
                        );

                    return result;
                },
                (int depthIndex, IDataSet vd, IArtifactContainer mlpContainer) =>
                {
                    return
                        new Validation(
                            new MetricsAccuracyCalculator(
                                new HalfSquaredEuclidianDistance(),
                                vd),
                            new GridReconstructDrawer(
                                new MNISTVisualizer(),
                                vd,
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
                        new HalfSquaredEuclidianDistance(), 
                        new LinearLearningRate(lr, 0.99f),
                        1,
                        0.001f,
                        3,
                        0f,
                        -0.0025f);

                    return conf;
                },
                (clProvider) => new GPUBackpropagationFactory(
                    mlpContainerHelper),
                (clProvider) => new ForwardPropagationFactory(
                    new GPUPropagatorComponentConstructor(
                        clProvider
                        )),
                new LayerInfo(784, new RLUFunction()),
                new LayerInfo(1200, new RLUFunction()),
                new LayerInfo(1200, new RLUFunction()),
                new LayerInfo(2000, new RLUFunction())
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

        private static IDataSet GetTrain(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new MNISTDataItemLoader(
                "_MNIST_DATABASE/mnist/trainingset/",
                maxCountFilesInCategory,
                false,
                dataItemFactory,
                new DefaultNormalizer()
                );

            if (isNeedToGNormalize)
            {
                dataItemLoader.GNormalize();
            }

            if (isNeedToNormalize)
            {
                dataItemLoader.Normalize();
            }

            var iterationFactory = new DataIteratorFactory(
                );

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new ToAutoencoderDataItemTransformation(
                            dataItemFactory);
                });

            var trainDataSet = new DataSet(
                iterationFactory,
                itemTransformationFactory,
                dataItemLoader,
                0
                );

            return
                trainDataSet;
        }

        private static IDataSet GetValidation(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new MNISTDataItemLoader(
                "_MNIST_DATABASE/mnist/testset/",
                maxCountFilesInCategory,
                false,
                dataItemFactory,
                new DefaultNormalizer()
                );

            if (isNeedToGNormalize)
            {
                dataItemLoader.GNormalize();
            }

            if (isNeedToNormalize)
            {
                dataItemLoader.Normalize();
            }

            var iterationFactory = new DataIteratorFactory(
                );

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new ToAutoencoderDataItemTransformation(
                            dataItemFactory);
                });

            var validationDataSet = new DataSet(
                iterationFactory,
                itemTransformationFactory,
                dataItemLoader,
                0
                );

            return
                validationDataSet;
        }

    }
}
