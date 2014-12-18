using System;
using MyNN.Boltzmann.BeliefNetwork.Accuracy;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
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

namespace MyNNConsoleApp.RBM
{
    public class TrainRBM
    {
        public static void DoTrainLNRELU()
        {
            const int validationMaxCountFilesInCategory = 300;
            const int trainMaxCountFilesInCategory = 1000;

            var randomizer = new DefaultRandomizer(123);

            var validationData = GetValidation(
                validationMaxCountFilesInCategory,
                true,
                false
                );

            const int visibleNeuronCount = 784;
            const int hiddenNeuronCount = 500;

            var calculator = new LNRELUCalculator(
                visibleNeuronCount, 
                hiddenNeuronCount);

            var container = new FloatArrayContainer(
                randomizer,
                null,
                visibleNeuronCount, 
                hiddenNeuronCount);

            var algorithm = new CD(
                calculator,
                container);

            var reconstructor = new IsolatedImageReconstructor(
                validationData, 
                300, 
                28, 
                28);

            var extractor = new IsolatedFeatureExtractor(
                hiddenNeuronCount, 
                28, 
                28);

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(".", serialization);

            var rbmContainer = rootContainer.GetChildContainer(
                string.Format(
                    "rbm{0}",
                    DateTime.Now.ToString("yyyyMMddHHmmss")
                    ));

            var rbm = new MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.RBM(
                rbmContainer,
                container,
                algorithm,
                reconstructor,
                extractor
                );

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        new NoConvertDataItemTransformation(
                            );
                });

            var iterationFactory = new DataIteratorFactory(
                );

            var dataSetFactory = new DataSetFactory(
                iterationFactory,
                itemTransformationFactory
                );

            var dataItemFactory = new DataItemFactory();

            IDataItemLoader dataItemLoader = new MNISTDataItemLoader(
                "_MNIST_DATABASE/mnist/trainingset/",
                trainMaxCountFilesInCategory,
                false,
                dataItemFactory,
                new DefaultNormalizer()
                );
            dataItemLoader.GNormalize();

            dataItemLoader = new ShuffleDataItemLoader(
                randomizer,
                dataItemLoader
                );

            var trainDataProvider = new DataSetProvider(
                dataSetFactory,
                dataItemLoader
                );

            rbm.Train(
                trainDataProvider,
                validationData,
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

            var validationData = GetValidation(
                validationMaxCountFilesInCategory,
                false,
                true
                );

            const int visibleNeuronCount = 784;
            const int hiddenNeuronCount = 500;

            var calculator = new BBCalculator(randomizer, visibleNeuronCount, hiddenNeuronCount);

            var feCalculator = new FloatArrayFreeEnergyCalculator(
                visibleNeuronCount,
                hiddenNeuronCount);

            var container = new FloatArrayContainer(
                randomizer, 
                feCalculator,
                visibleNeuronCount, 
                hiddenNeuronCount);

            var algorithm = new CD(
                calculator,
                container);

            var reconstructor = new IsolatedImageReconstructor(
                validationData, 
                300, 
                28, 
                28);

            var extractor = new IsolatedFeatureExtractor(
                hiddenNeuronCount, 
                28, 
                28);

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(".", serialization);

            var rbmContainer = rootContainer.GetChildContainer(
                string.Format(
                    "rbm{0}",
                    DateTime.Now.ToString("yyyyMMddHHmmss")
                    ));

            var rbm = new MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.RBM(
                rbmContainer,
                container,
                algorithm,
                reconstructor,
                extractor
                );

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

            var iterationFactory = new DataIteratorFactory(
                );

            var dataSetFactory = new DataSetFactory(
                iterationFactory,
                itemTransformationFactory
                );

            IDataItemLoader dataItemLoader = new MNISTDataItemLoader(
                "_MNIST_DATABASE/mnist/trainingset/",
                trainMaxCountFilesInCategory,
                true,
                dataItemFactory,
                new DefaultNormalizer()
                );
            dataItemLoader.Normalize();

            dataItemLoader = new ShuffleDataItemLoader(
                randomizer,
                dataItemLoader
                );

            var trainDataProvider = new DataSetProvider(
                dataSetFactory,
                dataItemLoader
                );

            rbm.Train(
                trainDataProvider,
                validationData,
                new LinearLearningRate(0.01f, 0.99f),
                new AccuracyController(
                    0.1f,
                    2),
                10,
                1);
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
