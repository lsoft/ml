using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN;
using MyNN.Boosting.SAMMEBoosting;
using MyNN.Data.DataSetConverter;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.Drawer;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TestBackProp
    {
        public static void DoTest()
        {
            var cpuChooser = new IntelCPUDeviceChooser(true);
            var gpuChooser = new NvidiaOrAmdGPUDeviceChooser(true);

            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IBackpropagationEpocheTrainer> cpuTrainer = (clProvider, mlp, config) =>
            {
                return
                    new CPUBackpropagationEpocheTrainer(
                        VectorizationSizeEnum.NoVectorization,
                        mlp,
                        config,
                        clProvider
                        );
            };

            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IBackpropagationEpocheTrainer> gpuTrainer = (clProvider, mlp, config) =>
            {
                return
                    new GPUBackpropagationEpocheTrainer(
                        mlp,
                        config,
                        clProvider
                        );
            };

            const int batchSize = 5;
            const float regularizationFactor = 0f;//1e-2f;

            var g0 = DoTestPrivate(
                gpuChooser,
                gpuTrainer,
                1,
                regularizationFactor
                );

            var g1 = DoTestPrivate(
                gpuChooser,
                gpuTrainer,
                batchSize,
                regularizationFactor
                );

            var c0 = DoTestPrivate(
                cpuChooser,
                cpuTrainer,
                1,
                regularizationFactor
                );

            var c1 = DoTestPrivate(
                cpuChooser,
                cpuTrainer,
                batchSize,
                regularizationFactor
                );


            var d0 = Math.Abs(g0.PerItemError - c0.PerItemError);
            const double d0threshold = 0.00000006;

            Console.ForegroundColor = d0 <= d0threshold ? ConsoleColor.Yellow : ConsoleColor.Red;
            Console.WriteLine("ERROR 0: {0}", DoubleConverter.ToExactString(d0));

            var d1 = Math.Abs(g1.PerItemError - c1.PerItemError);
            const double d1threshold = 0.000001;

            Console.ForegroundColor = d1 <= d1threshold ? ConsoleColor.Yellow : ConsoleColor.Red;
            Console.WriteLine("ERROR 1: {0}", DoubleConverter.ToExactString(d1));

            Console.ResetColor();
        }

        private static IAccuracyRecord DoTestPrivate(
            IDeviceChooser deviceChooser,
            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IBackpropagationEpocheTrainer> epocheTrainerFunc,
            int batchSize,
            float regularizationFactor
            )
        {
            var randomizer = new DefaultRandomizer(123);

            var binarizator = new BinarizeDataSetConverter(
                randomizer);
            var toa = new ToAutoencoderDataSetConverter();

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                10
                );
            trainData = toa.Convert(binarizator.Convert(trainData));

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                10
                );
            validationData = toa.Convert(binarizator.Convert(validationData));

            var mlpfactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var serialization = new SerializationHelper();

            var rootContainer = new SavelessArtifactContainer(
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

            using (var clProvider = new CLProvider(deviceChooser, true))
            {
                var mlpName = string.Format(
                    "testmlp{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var mlp = mlpfactory.CreateMLP(
                    mlpName,
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
                        5000,
                        5000,
                        784
                    });

                var config = new LearningAlgorithmConfig(
                    new ConstLearningRate(1f / 65536),
                    batchSize,
                    regularizationFactor,
                    1,
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
                    epocheTrainerFunc(
                        clProvider,
                        mlp,
                        config
                        ),
                    mlpContainer,
                    mlp,
                    validation,
                    config
                    );

                var result = algo.Train(
                    trainDataProvider
                    );

                return result;
            }
        }
    }
}
