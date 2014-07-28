using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.BeliefNetwork;
using MyNN.BeliefNetwork.Accuracy;
using MyNN.BeliefNetwork.ImageReconstructor;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator;
using MyNN.BoltzmannMachines;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.Data;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
using MyNN.Randomizer;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainDBN
    {
        public static void DoTrainLNRELU()
        {
            var randomizer = new DefaultRandomizer(123);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                //300
                int.MaxValue
                );
            trainData.GNormalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                //100
                int.MaxValue
                );
            validationData.GNormalize();

            var serializationHelper = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(".", serializationHelper);

            var dbnContainer = rootContainer.GetChildContainer(
                string.Format(
                    "dbn{0}",
                    DateTime.Now.ToString("yyyyMMddHHmmss")));

            var rbmFactory = new RBMLNRELUCDFactory(randomizer);

            var dbn = new DBN(
                randomizer,
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

            Func<IDataSet, ITrainDataProvider> trainDataProviderProvider =
                (layerTrainData) =>
                    new ConverterTrainDataProvider(
                        new ShuffleDataSetConverter(randomizer),
                        new NoDeformationTrainDataProvider(layerTrainData));

            dbn.Train(
                trainData,
                validationData,
                trainDataProviderProvider,
                featureExtractorFactory,
                stackedImageReconstructor,
                new LinearLearningRate(0.0001f, 0.99f),
                new AccuracyController(
                    0.1f,
                    120),
                10,
                1);
        }

        public static void DoTrainBB()
        {
            var randomizer = new DefaultRandomizer(123);

            var binarizer = new BinarizeDataSetConverter(randomizer);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                300
                //int.MaxValue
                );

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                100
                //int.MaxValue
                );
            validationData = binarizer.Convert(validationData);

            var serializationHelper = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(".", serializationHelper);

            var dbnContainer = rootContainer.GetChildContainer(
                string.Format(
                    "dbn{0}",
                    DateTime.Now.ToString("yyyyMMddHHmmss")));

            var rbmFactory = new RBMBBCDFactory(randomizer);

            var dbn = new DBN(
                randomizer,
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

            Func<IDataSet, ITrainDataProvider> trainDataProviderProvider =
                (layerTrainData) =>
                    new ConverterTrainDataProvider(
                        new ListDataSetConverter( 
                            new BinarizeDataSetConverter(randomizer),
                            new ShuffleDataSetConverter(randomizer)),
                    new NoDeformationTrainDataProvider(layerTrainData));

            dbn.Train(
                trainData,
                validationData,
                trainDataProviderProvider,
                featureExtractorFactory,
                stackedImageReconstructor,
                new LinearLearningRate(0.01f, 0.99f),
                new AccuracyController(
                    0.1f,
                    10),
                10,
                1);
        }

        private class IsolatedFeatureExtractorFactory : IFeatureExtractorFactory
        {
            public IFeatureExtractor CreateFeatureExtractor(
                int hiddenNeuronCount)
            {
                var featureExtractor = new IsolatedFeatureExtractor(
                    hiddenNeuronCount,
                    28,
                    28);

                return featureExtractor;
            }
        }

        private class RBMLNRELUCDFactory : IRBMFactory
        {
            private readonly IRandomizer _randomizer;

            public RBMLNRELUCDFactory(
                IRandomizer randomizer
                )
            {
                if (randomizer == null)
                {
                    throw new ArgumentNullException("randomizer");
                }

                _randomizer = randomizer;
            }

            public void CreateRBM(
                IDataSet trainData,
                IArtifactContainer rbmContainer,
                IImageReconstructor imageReconstructor,
                IFeatureExtractor featureExtractor,
                int visibleNeuronCount,
                int hiddenNeuronCount,
                out IRBM rbm,
                out IContainer container,
                out IAlgorithm algorithm)
            {
                if (trainData == null)
                {
                    throw new ArgumentNullException("trainData");
                }
                if (rbmContainer == null)
                {
                    throw new ArgumentNullException("rbmContainer");
                }
                if (imageReconstructor == null)
                {
                    throw new ArgumentNullException("imageReconstructor");
                }
                //featureExtractor allowed to be null

                var calculator = new LNRELUCalculator(
                    visibleNeuronCount,
                    hiddenNeuronCount);

                var facontainer = new FloatArrayContainer(
                    _randomizer,
                    null,
                    visibleNeuronCount,
                    hiddenNeuronCount);

                container = facontainer;

                algorithm = new CD(
                    calculator,
                    facontainer);

                rbm = new RBM(
                    rbmContainer,
                    _randomizer,
                    container,
                    algorithm,
                    imageReconstructor,
                    featureExtractor
                    );
            }
        }

        private class RBMBBCDFactory : IRBMFactory
        {
            private readonly IRandomizer _randomizer;

            public RBMBBCDFactory(
                IRandomizer randomizer
                )
            {
                if (randomizer == null)
                {
                    throw new ArgumentNullException("randomizer");
                }

                _randomizer = randomizer;
            }

            public void CreateRBM(
                IDataSet trainData,
                IArtifactContainer rbmContainer,
                IImageReconstructor imageReconstructor,
                IFeatureExtractor featureExtractor,
                int visibleNeuronCount,
                int hiddenNeuronCount,
                out IRBM rbm,
                out IContainer container,
                out IAlgorithm algorithm)
            {
                if (trainData == null)
                {
                    throw new ArgumentNullException("trainData");
                }
                if (rbmContainer == null)
                {
                    throw new ArgumentNullException("rbmContainer");
                }
                if (imageReconstructor == null)
                {
                    throw new ArgumentNullException("imageReconstructor");
                }
                //featureExtractor allowed to be null

                var calculator = new BBCalculator(
                    _randomizer, 
                    visibleNeuronCount, 
                    hiddenNeuronCount);

                var feCalculator = new FloatArrayFreeEnergyCalculator(
                    visibleNeuronCount,
                    hiddenNeuronCount);

                var facontainer = new FloatArrayContainer(
                    _randomizer,
                    feCalculator,
                    visibleNeuronCount,
                    hiddenNeuronCount);

                container = facontainer;

                algorithm = new CD(
                    calculator,
                    facontainer);

                rbm = new RBM(
                    rbmContainer,
                    _randomizer,
                    container,
                    algorithm,
                    imageReconstructor,
                    featureExtractor
                    );
            }
        }

    }
}
