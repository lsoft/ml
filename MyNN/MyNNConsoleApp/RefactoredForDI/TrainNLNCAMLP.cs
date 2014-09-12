using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.KNN;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.ClassificationMLP.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation1;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator.KNNTester;
using MyNN.MLP2.Backpropagation.Validation.NLNCA;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TrainNLNCAMLP
    {
        public static void DoTrain()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000//int.MaxValue
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300//int.MaxValue
                );
            validationData.Normalize();

            var randomizer = new DefaultRandomizer(123);

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var rootContainer = new FileSystemArtifactContainer(
                ".",
                serialization);

            var mlpName = string.Format(
                "nlncamlp{0}.mlp",
                DateTime.Now.ToString("yyyyMMddHHmmss"));

            var mlpContainer = rootContainer.GetChildContainer(mlpName);

            var validation = new Validation(
                new NLNCAAccuracyCalculator(
                    new DefaultKNNTester( 
                        new CPUOpenCLKNearestFactory(),
                        trainData,
                        validationData,
                        3),
                    validationData,
                    mlpContainer
                    ),
                new NLNCADrawer(
                    validationData,
                    mlpContainer,
                    new MNISTColorProvider()
                    )
                );

            using (var clProvider = new CLProvider())
            {
                var mlp = mlpfactory.CreateMLP(
                    mlpName,
                    new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f), 
                        new LinearFunction(1f), 
                    },
                    new int[]
                    {
                        784,
                        500,
                        2
                    });

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.5f, 0.98f),
                    400,
                    0f,
                    15,
                    -1f,
                    -1f
                    );

                var trainDataProvider =
                    new ConverterTrainDataProvider(
                        new ShuffleDataSetConverter(randomizer),
                        new NoDeformationTrainDataProvider(trainData)
                        );

                var algo = new BackpropagationAlgorithm(
                    new CPUNLNCABackpropagationEpocheTrainer(
                        VectorizationSizeEnum.VectorizationMode16,
                        mlp,
                        config,
                        clProvider,
                        (uzkii) => new DodfCalculatorOpenCL(
                            uzkii,
                            new VectorizedCpuDistanceDictCalculator())
                        ),
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
    }
}
