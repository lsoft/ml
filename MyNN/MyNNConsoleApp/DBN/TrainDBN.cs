using System;
using MyNN.Boltzmann;
using MyNN.Boltzmann.BeliefNetwork.Accuracy;
using MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.FeatureFactory;
using MyNN.Boltzmann.BeliefNetwork.ImageReconstructor;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Factory;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNNConsoleApp.DBN
{
    public class TrainDBN
    {
        //public static void DoTrainMLPOnAE()
        //{
        //    var dataItemFactory = new DataItemFactory();

        //    var trainData = MNISTDataLoader.GetDataSet(
        //        "_MNIST_DATABASE/mnist/trainingset/",
        //        int.MaxValue,
        //        true,
        //        dataItemFactory
        //        );
        //    trainData.GNormalize();

        //    var validationData = MNISTDataLoader.GetDataSet(
        //        "_MNIST_DATABASE/mnist/testset/",
        //        int.MaxValue,
        //        true,
        //        dataItemFactory
        //        );
        //    validationData.GNormalize();

        //    var randomizer = new DefaultRandomizer(123);

        //    var serialization = new SerializationHelper();

        //    var rootContainer = new FileSystemArtifactContainer(
        //        ".",
        //        serialization);

        //    var validation = new Validation(
        //        new ClassificationAccuracyCalculator(
        //            new HalfSquaredEuclidianDistance(),
        //            validationData),
        //        new GridReconstructDrawer(
        //            new MNISTVisualizer(),
        //            validationData,
        //            300,
        //            100)
        //        );


        //    using (var clProvider = new CLProvider())
        //    {
        //        var mlp = serialization.LoadFromFile<MLP>("aeondbn20140728201343.ae\\epoche 49\\aeondbn20140728201343.ae");
        //        mlp.AutoencoderCutTail();
        //        mlp.AddLayer(
        //            new SigmoidFunction(1f),
        //            10,
        //            false);

        //        var mlpName = string.Format(
        //            "mlponae{0}.mlp",
        //            DateTime.Now.ToString("yyyyMMddHHmmss"));

        //        mlp.OverwriteName(mlpName);

        //        var config = new LearningAlgorithmConfig(
        //            new HalfSquaredEuclidianDistance(), 
        //            new LinearLearningRate(0.006f, 0.99f),
        //            1,
        //            0f,
        //            50,
        //            -1f,
        //            -1f
        //            );

        //        var trainDataProvider =
        //            new ConverterTrainDataProvider(
        //                new ShuffleDataSetConverter(randomizer),
        //                new NoDeformationTrainDataProvider(trainData)
        //                );

        //        var mlpContainer = rootContainer.GetChildContainer(mlp.Name);

        //        var mlpContainerHelper = new MLPContainerHelper();

        //        var algo = new Backpropagation(
        //            new CPUEpocheTrainer(
        //                VectorizationSizeEnum.VectorizationMode16,
        //                mlp,
        //                config,
        //                clProvider),
        //            mlpContainerHelper,
        //            mlpContainer,
        //            mlp,
        //            validation,
        //            config
        //            );

        //        algo.Train(
        //            trainDataProvider
        //            );
        //    }
        //}

        //public static void DoTrainAutoencoder()
        //{
        //    var dataItemFactory = new DataItemFactory(); 
            
        //    var toa = new ToAutoencoderDataSetConverter(
        //        dataItemFactory);

        //    var trainData = MNISTDataLoader.GetDataSet(
        //        "_MNIST_DATABASE/mnist/trainingset/",
        //        int.MaxValue,
        //        true,
        //        dataItemFactory
        //        );
        //    trainData.GNormalize();
        //    trainData = toa.Convert(trainData);

        //    var validationData = MNISTDataLoader.GetDataSet(
        //        "_MNIST_DATABASE/mnist/testset/",
        //        int.MaxValue,
        //        true,
        //        dataItemFactory
        //        );
        //    validationData.GNormalize();
        //    validationData = toa.Convert(validationData);

        //    var randomizer = new DefaultRandomizer(123);

        //    var mlpfactory = new MLPFactory(
        //        new LayerFactory(
        //            new NeuronFactory(
        //                randomizer)));

        //    var serialization = new SerializationHelper();

        //    var rootContainer = new FileSystemArtifactContainer(
        //        ".",
        //        serialization);

        //    var validation = new Validation(
        //        new MetricsAccuracyCalculator(
        //            new HalfSquaredEuclidianDistance(),
        //            validationData),
        //        new GridReconstructDrawer(
        //            new MNISTVisualizer(),
        //            validationData,
        //            300,
        //            100)
        //        );

        //    const int epocheCount = 50;

        //    var config = new LearningAlgorithmConfig(
        //        new HalfSquaredEuclidianDistance(), 
        //        new LinearLearningRate(0.0001f, 0.99f),
        //        1,
        //        0f,
        //        epocheCount,
        //        -1f,
        //        -1f
        //        );

        //    var noiser = new SequenceNoiser(
        //        randomizer,
        //        true,
        //        new GaussNoiser(0.20f, false, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength)),
        //        new MultiplierNoiser(randomizer, 1f, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength)),
        //        new DistanceChangeNoiser(randomizer, 1f, 3, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength)),
        //        new SaltAndPepperNoiser(randomizer, 0.1f, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength)),
        //        new ZeroMaskingNoiser(randomizer, 0.25f, new RandomSeriesRange(randomizer, trainData.Data[0].InputLength))
        //        );

        //    var trainDataProvider =
        //        new ConverterTrainDataProvider(
        //            new ShuffleDataSetConverter(randomizer),
        //            new NoiseDataProvider(trainData, noiser, dataItemFactory)
        //            );

        //    var dbnInformation = new FileDBNInformation(
        //        "dbn20140728131850",
        //        serialization);

        //    var autoencoderName = string.Format(
        //        "aeondbn{0}.ae",
        //        DateTime.Now.ToString("yyyyMMddHHmmss"));

        //    var mlp = mlpfactory.CreateAutoencoderMLP(
        //        dbnInformation,
        //        autoencoderName,
        //        new IFunction[]
        //            {
        //                null,
        //                new RLUFunction(), 
        //                new RLUFunction(), 

        //                new RLUFunction(), 
                        
        //                new RLUFunction(), 
        //                new RLUFunction(), 
        //                new RLUFunction(), 
        //            });

        //    var mlpContainer = rootContainer.GetChildContainer(autoencoderName);

        //    var mlpContainerHelper = new MLPContainerHelper();

        //    using (var clProvider = new CLProvider())
        //    {
        //        var algo = new Backpropagation(
        //            new CPUEpocheTrainer(
        //                VectorizationSizeEnum.VectorizationMode16,
        //                mlp,
        //                config,
        //                clProvider),
        //            mlpContainerHelper,
        //            mlpContainer,
        //            mlp,
        //            validation,
        //            config);

        //        algo.Train(trainDataProvider);
        //    }
        //}

        //public static void DoTrainMLPOnDBN()
        //{
        //    var dataItemFactory = new DataItemFactory();

        //    var trainData = MNISTDataLoader.GetDataSet(
        //        "_MNIST_DATABASE/mnist/trainingset/",
        //        int.MaxValue,
        //        true,
        //        dataItemFactory
        //        );
        //    trainData.GNormalize();

        //    var validationData = MNISTDataLoader.GetDataSet(
        //        "_MNIST_DATABASE/mnist/testset/",
        //        int.MaxValue,
        //        true,
        //        dataItemFactory
        //        );
        //    validationData.GNormalize();

        //    var randomizer = new DefaultRandomizer(123);

        //    var mlpfactory = new MLPFactory(
        //        new LayerFactory(
        //            new NeuronFactory(
        //                randomizer)));

        //    var serialization = new SerializationHelper();

        //    var rootContainer = new FileSystemArtifactContainer(
        //        ".",
        //        serialization);

        //    var validation = new Validation(
        //        new ClassificationAccuracyCalculator(
        //            new HalfSquaredEuclidianDistance(),
        //            validationData),
        //        new GridReconstructDrawer(
        //            new MNISTVisualizer(),
        //            validationData,
        //            300,
        //            100)
        //        );

        //    var dbnInformation = new FileDBNInformation(
        //        "dbn20140728131850",
        //        serialization);

        //    using (var clProvider = new CLProvider())
        //    {
        //        var mlpName = string.Format(
        //            "mlpondbn{0}.mlp",
        //            DateTime.Now.ToString("yyyyMMddHHmmss"));

        //        var mlp = mlpfactory.CreateMLP(
        //            dbnInformation,
        //            mlpName,
        //            new IFunction[]
        //            {
        //                null,
        //                new RLUFunction(), 
        //                new RLUFunction(), 
        //                new RLUFunction(), 
        //                new SigmoidFunction(1f), 
        //            },
        //            new int[]
        //            {
        //                784,
        //                1000,
        //                500,
        //                500,
        //                10
        //            });

        //        var config = new LearningAlgorithmConfig(
        //            new HalfSquaredEuclidianDistance(), 
        //            new LinearLearningRate(0.006f, 0.99f),
        //            1,
        //            0f,
        //            50,
        //            -1f,
        //            -1f
        //            );

        //        var trainDataProvider =
        //            new ConverterTrainDataProvider(
        //                new ShuffleDataSetConverter(randomizer),
        //                new NoDeformationTrainDataProvider(trainData)
        //                );

        //        var mlpContainer = rootContainer.GetChildContainer(mlpName);

        //        var mlpContainerHelper = new MLPContainerHelper();

        //        var algo = new Backpropagation(
        //            new CPUEpocheTrainer(
        //                VectorizationSizeEnum.VectorizationMode16,
        //                mlp,
        //                config,
        //                clProvider),
        //            mlpContainerHelper,
        //            mlpContainer,
        //            mlp,
        //            validation,
        //            config
        //            );

        //        algo.Train(
        //            trainDataProvider
        //            );
        //    }
        //}

        public static void DoTrainLNRELU()
        {
            const int validationMaxCountFilesInCategory = 300;
            const int trainMaxCountFilesInCategory = 1000;

            var randomizer = new DefaultRandomizer(123);

            var dataItemFactory = new DataItemFactory();

            var trainData = GetTrain(
                trainMaxCountFilesInCategory,
                true,
                false
                );

            var validationData = GetValidation(
                validationMaxCountFilesInCategory,
                true,
                false
                );

            var serializationHelper = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(".", serializationHelper);

            var dbnContainer = rootContainer.GetChildContainer(
                string.Format(
                    "dbn{0}",
                    DateTime.Now.ToString("yyyyMMddHHmmss")));

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new NoConvertDataItemTransformation();
                });

            var dataIterationFactory = new DataIteratorFactory(
                );

            var dataSetFactory = new DataSetFactory(
                dataIterationFactory,
                itemTransformationFactory
                );

            var rbmFactory = new RBMLNRELUCDFactory(
                randomizer,
                dataItemFactory,
                dataSetFactory
                );

            var dbn = new MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.DBN(
                dbnContainer,
                rbmFactory,
                784,
                1000,
                500,
                500
                );

            var isolatedImageReconstructor = new IsolatedImageReconstructor(
                validationData,
                300,
                28,
                28);

            var stackedImageReconstructor = new StackedImageReconstructor(
                isolatedImageReconstructor);

            var featureExtractorFactory = new IsolatedFeatureExtractorFactory();

            Func<IDataSet, IDataSetProvider> trainDataProviderProvider =
                (layerTrainData) =>
                {
                    var iterationFactory2 = new DataIteratorFactory(
                        );

                    var itemTransformationFactory2 = new DataItemTransformationFactory(
                        (epochNumber) =>
                        {
                            return
                                new NoConvertDataItemTransformation();
                        });

                    var dataSetFactory2 = new DataSetFactory(
                        iterationFactory2,
                        itemTransformationFactory2
                        );

                    var result =  new DataSetProvider(
                        dataSetFactory2,
                        new ShuffleDataItemLoader(
                            randomizer,
                            new FromArrayDataItemLoader(
                                layerTrainData,
                                new DefaultNormalizer()))
                        );

                    return result;

                    //new ConverterTrainDataProvider(
                    //    new ShuffleDataSetConverter(randomizer),
                    //    new NoDeformationTrainDataProvider(layerTrainData));
                };

            dbn.Train(
                trainData,
                validationData,
                trainDataProviderProvider,
                featureExtractorFactory,
                stackedImageReconstructor,
                new LinearLearningRate(0.001f, 0.99f),
                new AccuracyController(
                    0.1f,
                    2),
                10,
                1);
        }

        public static void DoTrainBB()
        {
            const int validationMaxCountFilesInCategory = 300;
            const int trainMaxCountFilesInCategory = 1000;

            var randomizer = new DefaultRandomizer(123);

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

            var serializationHelper = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(".", serializationHelper);

            var dbnContainer = rootContainer.GetChildContainer(
                string.Format(
                    "dbn{0}",
                    DateTime.Now.ToString("yyyyMMddHHmmss")));

            var dataItemFactory = new DataItemFactory();

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new BinarizeDataItemTransformation(
                            randomizer,
                            dataItemFactory
                            );
                });

            var dataIterationFactory = new DataIteratorFactory(
                );

            var dataSetFactory = new DataSetFactory(
                dataIterationFactory,
                itemTransformationFactory
                );

            var rbmFactory = new RBMBBCDFactory(
                randomizer,
                dataItemFactory,
                dataSetFactory
                );

            var dbn = new MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.DBN(
                dbnContainer,
                rbmFactory,
                784,
                500,
                200
                );

            var isolatedImageReconstructor = new IsolatedImageReconstructor(
                validationData,
                300,
                28,
                28);

            var stackedImageReconstructor = new StackedImageReconstructor(
                isolatedImageReconstructor);

            var featureExtractorFactory = new IsolatedFeatureExtractorFactory();

            Func<IDataSet, IDataSetProvider> trainDataProviderProvider =
                (layerTrainData) =>
                {
                    var iterationFactory2 = new DataIteratorFactory(
                        );

                    var itemTransformationFactory2 = new DataItemTransformationFactory(
                        (epochNumber) =>
                        {
                            return
                                new NoConvertDataItemTransformation();
                        });

                    var dataSetFactory2 = new DataSetFactory(
                        iterationFactory2,
                        itemTransformationFactory2
                        );

                    var result = new DataSetProvider(
                        dataSetFactory2,
                        new ShuffleDataItemLoader(
                            randomizer,
                            new FromArrayDataItemLoader(
                                layerTrainData,
                                new DefaultNormalizer()))
                        );

                    return result;

                    //new ConverterTrainDataProvider(
                    //    new ListDataSetConverter( 
                    //        new BinarizeDataSetConverter(randomizer, dataItemFactory),
                    //        new ShuffleDataSetConverter(randomizer)),
                    //new NoDeformationTrainDataProvider(layerTrainData));
                };

            dbn.Train(
                trainData,
                validationData,
                trainDataProviderProvider,
                featureExtractorFactory,
                stackedImageReconstructor,
                new LinearLearningRate(0.01f, 0.99f),
                new AccuracyController(
                    0.1f,
                    2),
                10,
                1);
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
                true,
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
                        new NoConvertDataItemTransformation();
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
                true,
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
                        new NoConvertDataItemTransformation();
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
