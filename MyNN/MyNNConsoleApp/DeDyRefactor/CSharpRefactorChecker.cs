using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.MNIST;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer.Factory;
using MyNN.MLP.Classic.BackpropagationFactory.Classic.CSharp;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNNConsoleApp.DeDyRefactor
{
    public class CSharpRefactorChecker
    {
        public static void DoTrain()
        {
            var randomizer = new DefaultRandomizer(21425);

            const int trainMaxCountFilesInCategory = 100;
            const int validationMaxCountFilesInCategory = 30;

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

            var mlpFactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var mlp = mlpFactory.CreateMLP(
                DateTime.Now.ToString("yyyyMMddHHmmss") + ".mlp",
                new IFunction[]
                {
                    null,
                    new LinearFunction(1f),
                    new LinearFunction(1f),
                    new LinearFunction(1f),
                },
                new int[]
                {
                    784,
                    200,
                    500,
                    10
                }
                );

            ConsoleAmbientContext.Console.WriteLine("MLP " + mlp.GetLayerInformation());

            var rootContainer = new SavelessArtifactContainer(
                ".",
                serialization);

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new HalfSquaredEuclidianDistance(),
                    validationData),
                new GridReconstructDrawerFactory(
                    new MNISTVisualizerFactory(),
                    validationData,
                    300
                    )
                );

            const int epocheCount = 2;

            var config = new LearningAlgorithmConfig(
                new HalfSquaredEuclidianDistance(),
                new LinearLearningRate(0.02f, 0.99f),
                5,
                0.001f,
                epocheCount,
                -1f
                );

            var mlpContainer = rootContainer.GetChildContainer(mlp.Name);

            var mlpContainerHelper = new MLPContainerHelper();

            var backpropagationFactory = new CSharpBackpropagationFactory(
                mlpContainerHelper
                );

            var algo = backpropagationFactory.CreateBackpropagation(
                randomizer,
                mlpContainer,
                mlp,
                validation,
                config
                );

            var r = algo.Train(
                trainDataSetProvider
                );

            var ce = r.PerItemError;
            var de = 0.2993497550487518310546875;

            var diff = de - ce;

            if (Math.Abs(diff) > 1e-9f)
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Green;
            }

            Console.WriteLine(
                "Diff {0}",
                diff
                );
        }

        private static IDataSetProvider GetTrainProvider(
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

            var iteratorFactory = new DataIteratorFactory(
                );

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new NoConvertDataItemTransformation();
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
