using System;
using MyNN;
using MyNN.BeliefNetwork.Accuracy;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
using MyNN.Randomizer;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainRBM
    {
        public static void DoTrainLNRELU()
        {
            var randomizer = new DefaultRandomizer(123);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000
                //int.MaxValue
                );
            trainData.GNormalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300//int.MaxValue
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
                randomizer,
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

            var binarizer = new BinarizeDataSetConverter(randomizer);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000
                //int.MaxValue
                );

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300//int.MaxValue
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
                randomizer,
                container,
                algorithm,
                reconstructor,
                extractor
                );

            var trainDataProvider = new ConverterTrainDataProvider(
                new ListDataSetConverter( 
                    new BinarizeDataSetConverter(randomizer),
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
