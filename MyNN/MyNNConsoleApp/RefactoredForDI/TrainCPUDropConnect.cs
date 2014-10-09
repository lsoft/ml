using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.Data.TypicalDataProvider;
using MyNN.Common.LearningRateController;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.CPU;
using MyNN.MLP.DropConnect.Inferencer;
using MyNN.MLP.DropConnect.WeightMask.Factory;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    internal class TrainCPUDropConnect
    {
        public static void DoTrain()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000 //int.MaxValue
                );
            trainData.Normalize();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                300 //int.MaxValue
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

            var mlpContainerHelper = new MLPContainerHelper();

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

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(true), false))
            {
                var mlpName = string.Format(
                    "mlp{0}.dropconnect.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var mlp = mlpfactory.CreateMLP(
                    mlpName,
                    new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f),
                        new SigmoidFunction(1f),
                        new SigmoidFunction(1f),
                    },
                    new int[]
                    {
                        784,
                        800,
                        800,
                        10
                    });

                var config = new LearningAlgorithmConfig(
                    new LinearLearningRate(0.2f, 0.99f),
                    1,
                    0f,
                    500,
                    -1f,
                    -1f
                    );

                var trainDataProvider =
                    new ConverterTrainDataProvider(
                        new ShuffleDataSetConverter(randomizer),
                        new NoDeformationTrainDataProvider(trainData)
                        );

                var mlpContainer = rootContainer.GetChildContainer(mlpName);


                const int sampleCount = 10000;
                const float p = 0.5f;

                var maskContainerFactory = new BigArrayWeightMaskContainerFactory(
                    randomizer,
                    clProvider);

                var inferencerFactory = new CPULayerInferencerFactory(
                    randomizer,
                    clProvider,
                    sampleCount,
                    p);

                var algo = new BackpropagationAlgorithm(
                    new DropConnectEpocheTrainer(
                        mlp,
                        config,
                        maskContainerFactory,
                        inferencerFactory,
                        clProvider,
                        sampleCount,
                        p),
                    mlpContainerHelper,
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
