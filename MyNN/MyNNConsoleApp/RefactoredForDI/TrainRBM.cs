using System;
using MyNN;
using MyNN.Boltzmann.BeliefNetwork.Accuracy;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set.Item.Dense;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TypicalDataProvider;
using MyNN.Common.LearningRateController;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainRBM
    {
        public static void DoTrainLNRELU()
        {
            var randomizer = new DefaultRandomizer(123);

            var dataItemFactory = new DenseDataItemFactory();

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000,
                true,
                dataItemFactory
                );
            trainData.GNormalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300,
                true,
                dataItemFactory
                );
            validationData.GNormalize();

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

            var rbm = new RBM(
                rbmContainer,
                container,
                algorithm,
                reconstructor,
                extractor
                );

            var trainDataProvider = new ConverterTrainDataProvider(
                new ShuffleDataSetConverter(randomizer),
                new NoDeformationTrainDataProvider(trainData)
                );

            rbm.Train(
                trainDataProvider,
                validationData,
                new LinearLearningRate(0.0001f, 0.99f), 
                new AccuracyController(
                    0.1f,
                    20),
                10,
                1);
        }

        public static void DoTrainBB()
        {
            var randomizer = new DefaultRandomizer(123);

            var dataItemFactory = new DenseDataItemFactory(); 
            
            var binarizer = new BinarizeDataSetConverter(
                randomizer,
                dataItemFactory
                );

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000,
                true,
                dataItemFactory
                );

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300,
                true,
                dataItemFactory
                );
            validationData = binarizer.Convert(validationData);

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

            var rbm = new RBM(
                rbmContainer,
                container,
                algorithm,
                reconstructor,
                extractor
                );

            var trainDataProvider = new ConverterTrainDataProvider(
                new ListDataSetConverter( 
                    new BinarizeDataSetConverter(randomizer, dataItemFactory),
                    new ShuffleDataSetConverter(randomizer)),
                new NoDeformationTrainDataProvider(trainData)
                );

            rbm.Train(
                trainDataProvider,
                validationData,
                new LinearLearningRate(0.01f, 0.99f),
                new AccuracyController(
                    0.1f,
                    20),
                10,
                1);
        }
    }
}
