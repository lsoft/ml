﻿using System;
using System.IO;
using System.Security.Permissions;
using System.Windows.Forms;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.DataLoader;

using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Data.TrainDataProvider.Noiser.Range;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TuneSDAE
    {
        public static void Tune()
        {
            const int trainMaxCountFilesInCategory = 1000;
            const int validationMaxCountFilesInCategory = 300;

            var trainDataSetProvider = GetTrainProvider(
                trainMaxCountFilesInCategory,
                false,
                true
                );

            var validationData = GetValidation(
                validationMaxCountFilesInCategory,
                false,
                true
                );

            var serialization = new SerializationHelper();

            string filepath = null;
            using (var ofd = new OpenFileDialog())
            {
                ofd.InitialDirectory = Directory.GetCurrentDirectory();

                ofd.Multiselect = false;

                if (ofd.ShowDialog() != DialogResult.OK)
                {
                    return;
                }

                filepath = ofd.FileNames[0];
            }

            var mlp = new SerializationHelper().LoadFromFile<MLP>(
                filepath
                );

            Console.WriteLine("Network configuration: " + mlp.GetLayerInformation());


            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(true), false))
            {
                const int epocheCount = 250;

                var config = new LearningAlgorithmConfig(
                    new HalfSquaredEuclidianDistance(),
                    new LinearLearningRate(0.001f, 0.99f),
                    1,
                    0.001f,
                    epocheCount,
                    0.0001f,
                    -1.0f);

                var rootContainer = new FileSystemArtifactContainer(
                    ".",
                    serialization);

                var sdaeName = string.Format(
                    "sdae{0}.tuned.sdae",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var mlpContainer = rootContainer.GetChildContainer(
                    sdaeName);

                var mlpContainerHelper = new MLPContainerHelper();

                var validation = new Validation(
                    new MetricsAccuracyCalculator(
                        new HalfSquaredEuclidianDistance(),
                        validationData),
                    new GridReconstructDrawer(
                        new MNISTVisualizer(),
                        validationData,
                        300,
                        100)
                    );

                var alg =
                    new Backpropagation(
                        new GPUEpocheTrainer(
                            mlp,
                            config,
                            clProvider),
                        mlpContainerHelper,
                        mlpContainer,
                        mlp,
                        validation,
                        config);

                //var noiser = new SequenceNoiser(
                //    randomizer,
                //    true,
                //    new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer)),
                //    new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer)),
                //    new DistanceChangeNoiser(randomizer, 1f, 3, new RandomSeriesRange(randomizer)),
                //    new SaltAndPepperNoiser(randomizer, 0.1f, new RandomSeriesRange(randomizer)),
                //    new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer))
                //    );

                //var iteratorFactory2 = new DataIteratorFactory();

                //var itemTransformationFactory2 = new DataItemTransformationFactory(
                //    (epochNumber) =>
                //    {
                //        return
                //            new NoiserDataItemTransformation(
                //                dataItemFactory,
                //                epochNumber,
                //                noiser,
                //                null
                //                );
                //    });


                //var dataSetFactory2 = new DataSetFactory(
                //    iteratorFactory2,
                //    itemTransformationFactory2
                //    );

                //var dataSetProvider = new DataSetProvider(
                //    dataSetFactory2,
                //    new FromArrayDataItemLoader(
                //        trainData,
                //        new DefaultNormalizer())
                //    );

                //обучение сети
                alg.Train(
                    trainDataSetProvider);
            }
        }


        private static IDataSetProvider GetTrainProvider(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var rndSeed = 81262;

            var randomizer = new DefaultRandomizer(++rndSeed);

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

            var noiser = new SequenceNoiser(
                randomizer,
                true,
                new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer)),
                new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomSeriesRange(randomizer)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomSeriesRange(randomizer)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer))
                );

            var iteratorFactory = new DataIteratorFactory();

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    var result = 
                        new ListDataItemTransformation(
                            new ToAutoencoderDataItemTransformation(
                                dataItemFactory),
                            new NoiserDataItemTransformation(
                                dataItemFactory,
                                epochNumber,
                                noiser,
                                null
                                )
                            );

                    return
                        result;
                });

            var dataSetFactory = new DataSetFactory(
                iteratorFactory,
                itemTransformationFactory
                );

            var dataSetProvider = new DataSetProvider(
                dataSetFactory,
                dataItemLoader
                );

            return
                dataSetProvider;
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
