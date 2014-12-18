using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.CSharp;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp
{
    public class CompareBP
    {
        public static void DoCompare()
        {
            Do(true);
            Do(false); 
        }

        private static void Do(
            bool userOpencl
            )
        {
            var randomizer = 
                new NoRandomRandomizer();
                //new DefaultRandomizer(2387184);

            var trainDataSetProvider = GetTrainProvider(
                100,
                false,
                false
                );
            
            var validationData = GetValidation(
                100,
                false,
                false
                );

            var serialization = new SerializationHelper();

            var mlpFactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var mlp = mlpFactory.CreateMLP(
                string.Format(
                    "mlp{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss")),
                new IFunction[]
                {
                    null,
                    //new LinearFunction(1f), 
                    new LinearFunction(1f), 
                },
                new int[]
                {
                    validationData.InputLength,
                    //100,
                    validationData.OutputLength
                });

            ConsoleAmbientContext.Console.WriteLine("Created " + mlp.GetLayerInformation());

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new HalfSquaredEuclidianDistance(), 
                    validationData), 
                null
                );

            const int epocheCount = 1;

            var config = new LearningAlgorithmConfig(
                new HalfSquaredEuclidianDistance(),
                new ConstLearningRate(0.02f), 
                1,
                0f,
                epocheCount,
                -1f,
                -1f
                );

            var mlpFolderName = string.Format(
                "mlp{0}",
                DateTime.Now.ToString("yyyyMMddHHmmss"));

            var mlpContainer = rootContainer.GetChildContainer(mlpFolderName);

            var mlpContainerHelper = new MLPContainerHelper();


            if (userOpencl)
            {
                using (var clProvider = new CLProvider())
                {
                    var algo = new Backpropagation(
                        new CPUEpocheTrainer(
                            VectorizationSizeEnum.NoVectorization,
                            mlp,
                            config,
                            clProvider),
                        mlpContainerHelper,
                        mlpContainer,
                        mlp,
                        validation,
                        config
                        );

                    algo.Train(
                        trainDataSetProvider
                        );
                }
            }
            else
            {
                var algo = new Backpropagation(
                    new CSharpEpocheTrainer(
                        mlp,
                        config),
                    mlpContainerHelper,
                    mlpContainer,
                    mlp,
                    validation,
                    config
                    );

                algo.Train(
                    trainDataSetProvider
                    );
            }
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

            var iteratorFactory = new DataIteratorFactory();

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
