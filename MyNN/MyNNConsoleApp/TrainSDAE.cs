using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.Set.Item.Lazy;
using MyNN.Common.Data.Set.Item.Sparse;
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

namespace MyNNConsoleApp
{
    public class TrainSDAE
    {
        public static void DoTrain()
        {
            var combinedData = SAMRDataProvider.GetDataSet(
                "..\\..\\..\\..\\..\\data #1\\bin\\combineddata.bin",
                int.MaxValue
                );

            {
                var randomizer2 = new DefaultRandomizer(898989);

                for (int i = 0; i < combinedData.Count - 1; i++)
                {
                    if (randomizer2.Next() >= 0.5d)
                    {
                        var newIndex = randomizer2.Next(combinedData.Count);

                        var tmp = combinedData[i];
                        combinedData[i] = combinedData[newIndex];
                        combinedData[newIndex] = tmp;
                    }
                }

            }

            combinedData = combinedData.Take(100000).ToList();

            GC.Collect();

            const int ItemSize = 17525;

            var trainpart = (int)(0.9f * combinedData.Count);

            var trainData = new DataSet(combinedData.Take(trainpart).ToList());
            var validationData = new DataSet(combinedData.Skip(trainpart).ToList());

            var randomizer = new DefaultRandomizer(123);

            var serialization = new SerializationHelper();

            var sparseDataItemFactory = new SparseDataItemFactory();
            Func<INoiser, IDataItemFactory> lazyDataItemFactoryFunc = 
                (INoiser noiser) =>
                    new LazyDataItemFactory(
                        noiser,
                        serialization
                        );

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var toa = new ToAutoencoderDataSetConverter(
                sparseDataItemFactory
                );

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var oneDepthNoiser = new SequenceNoiser(
                randomizer,
                true,
                new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer, trainData[0].InputLength)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer, trainData[0].InputLength))
                );

            var deepDepthNoiser = new SequenceNoiser(
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
                    sparseDataItemFactory,
                    mlpfactory,
                    (int depthIndex, IDataSet td) =>
                    {
                        var tda = toa.Convert(td);

                        var result =
                            new ConverterTrainDataProvider(
                                new ShuffleDataSetConverter(randomizer),
                                new LazyNoiseDataProvider(
                                    tda, 
                                    depthIndex == 0
                                        ? oneDepthNoiser
                                        : deepDepthNoiser,
                                   lazyDataItemFactoryFunc)
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
                                null
                                );
                    },
                    (int depthIndex) =>
                    {
                        var lr =
                            depthIndex == 0
                                ? 0.001f
                                : 0.001f;

                        var conf = new LearningAlgorithmConfig(
                            new LinearLearningRate(lr, 0.99f),
                            10,
                            0.0f,
                            25,
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
                    new LayerInfo(ItemSize, new RLUFunction()),
                    new LayerInfo(3000, new RLUFunction()),
                    new LayerInfo(3000, new RLUFunction()),
                    new LayerInfo(6000, new RLUFunction())
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
