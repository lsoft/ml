using System;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set.Item.Dense;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TypicalDataProvider;
using MyNN.Common.LearningRateController;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.EpocheTrainer;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.GPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.CPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.GPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic2.OpenCL.CPU;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.TransposedClassic2.OpenCL.GPU;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
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

            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IEpocheTrainer> cpuTrainer = (clProvider, mlp, config) =>
            {
                return
                    new CPUEpocheTrainer(
                        VectorizationSizeEnum.NoVectorization,
                        mlp,
                        config,
                        clProvider
                        );
            };

            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IEpocheTrainer> gpuTrainer = (clProvider, mlp, config) =>
            {
                return
                    new GPUEpocheTrainer(
                        mlp,
                        config,
                        clProvider
                        );
            };

            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IEpocheTrainer> cpuTransposeTrainer = (clProvider, mlp, config) =>
            {
                return
                    new CPUTransposeEpocheTrainer(
                        VectorizationSizeEnum.NoVectorization,
                        mlp,
                        config,
                        clProvider
                        );
            };

            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IEpocheTrainer> gpuTransposeTrainer = (clProvider, mlp, config) =>
            {
                return
                    new GPUTransposeEpocheTrainer(
                        mlp,
                        config,
                        clProvider
                        );
            };

            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IEpocheTrainer> cpuTranspose2Trainer = (clProvider, mlp, config) =>
            {
                return
                    new CPUTranspose2EpocheTrainer(
                        VectorizationSizeEnum.NoVectorization, 
                        mlp,
                        config,
                        clProvider
                        );
            };

            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IEpocheTrainer> gpuTranspose2Trainer = (clProvider, mlp, config) =>
            {
                return
                    new GPUTranspose2EpocheTrainer(
                        mlp,
                        config,
                        clProvider
                        );
            };

            const int batchSize = 5;
            const float regularizationFactor = 0f;//1e-2f;

            var g0 = DoTestPrivate(
                gpuChooser,
                gpuTranspose2Trainer,
                1,
                regularizationFactor
                );

            var g1 = DoTestPrivate(
                gpuChooser,
                gpuTranspose2Trainer,
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
            Func<CLProvider, IMLP, ILearningAlgorithmConfig, IEpocheTrainer> epocheTrainerFunc,
            int batchSize,
            float regularizationFactor
            )
        {
            var randomizer = new DefaultRandomizer(123);

            var dataItemFactory = new DenseDataItemFactory();

            var binarizator = new BinarizeDataSetConverter(
                randomizer,
                dataItemFactory
                )
                ;
            var toa = new ToAutoencoderDataSetConverter(
                dataItemFactory);

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                10,
                true,
                dataItemFactory
                );
            trainData = toa.Convert(binarizator.Convert(trainData));

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                10,
                true,
                dataItemFactory
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

                var mlpContainerHelper = new MLPContainerHelper();

                var algo = new Backpropagation(
                    epocheTrainerFunc(
                        clProvider,
                        mlp,
                        config
                        ),
                    mlpContainerHelper,
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
