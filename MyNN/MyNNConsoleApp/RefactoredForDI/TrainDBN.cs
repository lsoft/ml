﻿using System;
using System.Collections.Generic;
using System.Drawing;
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
using MyNN.BoltzmannMachines.DBNInfo;
using MyNN.Data;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.Data.Visualizer;
using MyNN.LearningRateController;
using MyNN.MLP2;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainDBN
    {
        public static void DoTrainMLPOnAE()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                );
            trainData.GNormalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                );
            validationData.GNormalize();

            var randomizer = new DefaultRandomizer(123);

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new HalfSquaredEuclidianDistance(),
                    validationData),
                new GridReconstructDrawer(
                    new MNISTVisualizer(),
                    validationData,
                    300,
                    100)
                );


            using (var clProvider = new CLProvider())
            {
                var mlp = serialization.LoadFromFile<MLP>("aeondbn20140728201343.ae\\epoche 49\\aeondbn20140728201343.ae");
                mlp.AutoencoderCutTail();
                mlp.AddLayer(
                    new SigmoidFunction(1f),
                    10,
                    false);

                var mlpName = string.Format(
                    "mlponae{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                mlp.OverwriteName(mlpName);

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.006f, 0.99f),
                    1,
                    0f,
                    50,
                    -1f,
                    -1f
                    );

                var trainDataProvider =
                    new ConverterTrainDataProvider(
                        new ShuffleDataSetConverter(randomizer),
                        new NoDeformationTrainDataProvider(trainData)
                        );

                var mlpContainer = rootContainer.GetChildContainer(mlp.Name);

                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    new CPUBackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider),
                    mlpContainer,
                    mlp,
                    validation,
                    config
                    );

                algo.Train(
                    trainDataProvider
                    );
            }
        }

        public static void DoTrainAutoencoder()
        {
            var toa = new ToAutoencoderDataSetConverter();

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                //100
                int.MaxValue
                );
            trainData.GNormalize();
            trainData = toa.Convert(trainData);

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                //100
                int.MaxValue
                );
            validationData.GNormalize();
            validationData = toa.Convert(validationData);

            var randomizer = new DefaultRandomizer(123);

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

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

            const int epocheCount = 50;

            var config = new LearningAlgorithmConfig(
                new LinearLearningRate(0.0001f, 0.99f),
                1,
                0f,
                epocheCount,
                -1f,
                -1f
                );

            var noiser = new AllNoisers(
                randomizer,
                new GaussNoiser(0.20f, false, new RandomRange(randomizer)),
                new MultiplierNoiser(randomizer, 1f, new RandomRange(randomizer)),
                new DistanceChangeNoiser(randomizer, 1f, 3, new RandomRange(randomizer)),
                new SaltAndPepperNoiser(randomizer, 0.1f, new RandomRange(randomizer)),
                new ZeroMaskingNoiser(randomizer, 0.25f, new RandomRange(randomizer))
                );

            var trainDataProvider =
                new ConverterTrainDataProvider(
                    new ShuffleDataSetConverter(randomizer),
                    new NoiseDataProvider(trainData, noiser)
                    );

            var dbnInformation = new FileDBNInformation(
                "dbn20140728131850",
                serialization);

            var autoencoderName = string.Format(
                "aeondbn{0}.ae",
                DateTime.Now.ToString("yyyyMMddHHmmss"));

            var mlp = mlpfactory.CreateAutoencoderMLP(
                dbnInformation,
                autoencoderName,
                new IFunction[]
                    {
                        null,
                        new RLUFunction(), 
                        new RLUFunction(), 

                        new RLUFunction(), 
                        
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                    });

            var mlpContainer = rootContainer.GetChildContainer(autoencoderName);

            using (var clProvider = new CLProvider())
            {
                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    new CPUBackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider),
                    mlpContainer,
                    mlp,
                    validation,
                    config);

                algo.Train(trainDataProvider);
            }
        }

        public static void DoTrainMLPOnDBN()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                int.MaxValue
                );
            trainData.GNormalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                int.MaxValue
                );
            validationData.GNormalize();

            var randomizer = new DefaultRandomizer(123);

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new HalfSquaredEuclidianDistance(),
                    validationData),
                new GridReconstructDrawer(
                    new MNISTVisualizer(),
                    validationData,
                    300,
                    100)
                );

            var dbnInformation = new FileDBNInformation(
                "dbn20140728131850",
                serialization);

            using (var clProvider = new CLProvider())
            {
                var mlpName = string.Format(
                    "mlpondbn{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var mlp = mlpfactory.CreateMLP(
                    dbnInformation,
                    mlpName,
                    new IFunction[]
                    {
                        null,
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new RLUFunction(), 
                        new SigmoidFunction(1f), 
                    },
                    new int[]
                    {
                        784,
                        1000,
                        500,
                        500,
                        10
                    });

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.006f, 0.99f),
                    1,
                    0f,
                    50,
                    -1f,
                    -1f
                    );

                var trainDataProvider =
                    new ConverterTrainDataProvider(
                        new ShuffleDataSetConverter(randomizer),
                        new NoDeformationTrainDataProvider(trainData)
                        );

                var mlpContainer = rootContainer.GetChildContainer(mlpName);

                var algo = new BackpropagationAlgorithm(
                    randomizer,
                    new CPUBackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider),
                    mlpContainer,
                    mlp,
                    validation,
                    config
                    );

                algo.Train(
                    trainDataProvider
                    );
            }
        }

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
                    60),
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
                out IDataSetConverter forwardDataSetConverter,
                out IDataArrayConverter dataArrayConverter
                )
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

                var container = new FloatArrayContainer(
                    _randomizer,
                    null,
                    visibleNeuronCount,
                    hiddenNeuronCount);

                var algorithm = new CD(
                    calculator,
                    container);

                rbm = new RBM(
                    rbmContainer,
                    _randomizer,
                    container,
                    algorithm,
                    imageReconstructor,
                    featureExtractor
                    );

                forwardDataSetConverter = new ForwardDataSetConverter(
                    container,
                    algorithm);

                dataArrayConverter = new ImageReconstructorDataConverter(
                    container,
                    algorithm);
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
                out IDataSetConverter forwardDataSetConverter,
                out IDataArrayConverter dataArrayConverter
                )
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

                var container = new FloatArrayContainer(
                    _randomizer,
                    feCalculator,
                    visibleNeuronCount,
                    hiddenNeuronCount);


                var algorithm = new CD(
                    calculator,
                    container);

                rbm = new RBM(
                    rbmContainer,
                    _randomizer,
                    container,
                    algorithm,
                    imageReconstructor,
                    featureExtractor
                    );

                forwardDataSetConverter = new ForwardDataSetConverter(
                    container,
                    algorithm);

                dataArrayConverter = new ImageReconstructorDataConverter(
                    container,
                    algorithm);
            }
        }

        private class ImageReconstructorDataConverter : IDataArrayConverter
        {
            private readonly IContainer _container;
            private readonly IAlgorithm _algorithm;

            public ImageReconstructorDataConverter(
                IContainer container,
                IAlgorithm algorithm)
            {
                if (container == null)
                {
                    throw new ArgumentNullException("container");
                }
                if (algorithm == null)
                {
                    throw new ArgumentNullException("algorithm");
                }

                _container = container;
                _algorithm = algorithm;
            }

            public float[] Convert(float[] dataToConvert)
            {
                if (dataToConvert == null)
                {
                    throw new ArgumentNullException("dataToConvert");
                }

                _container.SetHidden(dataToConvert);
                
                var result = _algorithm.CalculateVisible();
                
                return result;
            }
        }

        private class ForwardDataSetConverter : IDataSetConverter
        {
            private readonly IContainer _container;
            private readonly IAlgorithm _algorithm;

            public ForwardDataSetConverter(
                IContainer container,
                IAlgorithm algorithm)
            {
                if (container == null)
                {
                    throw new ArgumentNullException("container");
                }
                if (algorithm == null)
                {
                    throw new ArgumentNullException("algorithm");
                }

                _container = container;
                _algorithm = algorithm;
            }

            public IDataSet Convert(IDataSet beforeTransformation)
            {
                if (beforeTransformation == null)
                {
                    throw new ArgumentNullException("beforeTransformation");
                }

                var newdiList = new List<DataItem>();
                foreach (var di in beforeTransformation)
                {
                    _container.SetInput(di.Input);
                    var nextLayer = _algorithm.CalculateHidden();

                    var newdi = new DataItem(
                        nextLayer,
                        di.Output);

                    newdiList.Add(newdi);
                }

                var result = new DataSet(newdiList);

                return result;
            }
        }
    }
}