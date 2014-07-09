using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagation;
using MyNN.MLP2.Backpropagation.EpocheTrainer.Classic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Float.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.TransposedClassic.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.EpocheTrainer.TransposedClassic2.OpenCL.CPU;
using MyNN.MLP2.Backpropagation.Metrics;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU.Inferencer;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Saver;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.DerivativeChange
{
    internal class Test0
    {
        public static void Execute()
        {
            BackpropagationOnline();
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);

            BackpropagationTranspose();
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);

            BackpropagationTranspose2();
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);

            BackpropagationDropConnectBit();
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
            Console.WriteLine(string.Empty);
        }

        private static void BackpropagationOnline()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1
                );
            trainData.Normalize();
            trainData = new DataSet(
                new List<DataItem>
                {
                    trainData.Data[0],
                    //trainData.Data[1],
                },
                trainData.Visualizer);

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                10
                );
            validationData.Normalize();

            {
                var randomizer =
                    new NoRandomRandomizer();

                var folderName = "_DerivativeMLP" + DateTime.Now.ToString("yyyyMMddHHmmss");

                var layerFactory = new LayerFactory(new NeuronFactory(randomizer));

                var mlpf = new MLPFactory(
                    layerFactory);

                var mlp = mlpf.CreateMLP(
                    null,
                    folderName,
                    new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f), 
                        new SigmoidFunction(1f), 
                    },
                        new int[]
                    {
                        784,
                        500,
                        10
                    });


                using (var clProvider = new CLProvider())
                {
                    var config = new LearningAlgorithmConfig(
                        new ConstLearningRate(0.1f),
                        1,
                        0.0f,
                        1,
                        0.0001f,
                        -1.0f);

                    var validation = new ClassificationValidation(
                        new ConsoleMLPSaver(), 
                        new HalfSquaredEuclidianDistance(),
                        validationData,
                        300,
                        100);

                    var alg =
                        new BackpropagationAlgorithm(
                            randomizer,
                            new CPUBackpropagationEpocheTrainer(
                                VectorizationSizeEnum.VectorizationMode16,
                                mlp,
                                config,
                                clProvider),
                            mlp,
                            validation,
                            config);

                    //обучение сети
                    alg.Train(
                        new NoDeformationTrainDataProvider(trainData));
                }
            }
        }

        private static void BackpropagationTranspose()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1
                );
            trainData.Normalize();
            trainData = new DataSet(
                new List<DataItem>
                {
                    trainData.Data[0],
                    trainData.Data[1],
                },
                trainData.Visualizer);

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                10
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            {
                var randomizer =
                    new NoRandomRandomizer();

                var folderName = "_DerivativeMLP" + DateTime.Now.ToString("yyyyMMddHHmmss");

                var layerFactory = new LayerFactory(new NeuronFactory(randomizer));

                var mlpf = new MLPFactory(
                    layerFactory);

                var mlp = mlpf.CreateMLP(
                    null,
                    folderName,
                    new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f), 
                        new SigmoidFunction(1f), 
                    },
                        new int[]
                    {
                        784,
                        500,
                        10
                    });


                using (var clProvider = new CLProvider())
                {
                    var config = new LearningAlgorithmConfig(
                        new ConstLearningRate(0.1f),
                        2,
                        0.0f,
                        1,
                        0.0001f,
                        -1.0f);

                    var validation = new ClassificationValidation(
                        new ConsoleMLPSaver(), 
                        new HalfSquaredEuclidianDistance(),
                        validationData,
                        300,
                        100);

                    var alg =
                        new BackpropagationAlgorithm(
                            randomizer,
                            new CPUTransposeBackpropagationEpocheTrainer(
                                VectorizationSizeEnum.VectorizationMode16,
                                mlp,
                                config,
                                clProvider),
                            mlp,
                            validation,
                            config);

                    //обучение сети
                    alg.Train(
                        new NoDeformationTrainDataProvider(trainData));
                }
            }
        }

        private static void BackpropagationTranspose2()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1
                );
            trainData.Normalize();
            trainData = new DataSet(
                new List<DataItem>
                {
                    trainData.Data[0],
                    trainData.Data[1],
                },
                trainData.Visualizer);

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                10
                );
            validationData.Normalize();

            var serialization = new SerializationHelper();

            {
                var randomizer =
                    new NoRandomRandomizer();

                var folderName = "_DerivativeMLP" + DateTime.Now.ToString("yyyyMMddHHmmss");

                var layerFactory = new LayerFactory(new NeuronFactory(randomizer));

                var mlpf = new MLPFactory(
                    layerFactory);

                var mlp = mlpf.CreateMLP(
                    null,
                    folderName,
                    new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f), 
                        new SigmoidFunction(1f), 
                    },
                        new int[]
                    {
                        784,
                        500,
                        10
                    });


                using (var clProvider = new CLProvider())
                {
                    var config = new LearningAlgorithmConfig(
                        new ConstLearningRate(0.1f),
                        2,
                        0.0f,
                        1,
                        0.0001f,
                        -1.0f);

                    var validation = new ClassificationValidation(
                        new ConsoleMLPSaver(), 
                        new HalfSquaredEuclidianDistance(),
                        validationData,
                        300,
                        100);

                    var alg =
                        new BackpropagationAlgorithm(
                            randomizer,
                            new CPUTranspose2BackpropagationEpocheTrainer(
                                VectorizationSizeEnum.VectorizationMode16,
                                mlp,
                                config,
                                clProvider),
                            mlp,
                            validation,
                            config);

                    //обучение сети
                    alg.Train(
                        new NoDeformationTrainDataProvider(trainData));
                }
            }
        }

        private static void BackpropagationDropConnectBit()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                            "_MNIST_DATABASE/mnist/trainingset/",
                            1
                            );
            trainData.Normalize();
            trainData = new DataSet(
                new List<DataItem>
                {
                    trainData.Data[0],
                    //trainData.Data[1],
                },
                trainData.Visualizer);

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                10
                );
            validationData.Normalize();

            const float p = 0.5f;
            const int sampleCount = 10000;

            var serialization = new SerializationHelper();

            {
                var randomizer =
                    new NoRandomRandomizer();

                var folderName = "_DropConnectMLP" + DateTime.Now.ToString("yyyMMddHHmmss");

                var layerFactory = new LayerFactory(new NeuronFactory(randomizer));

                var mlpf = new MLPFactory(
                    layerFactory);

                var mlp = mlpf.CreateMLP(
                    null,
                    folderName,
                    new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f), 
                        new SigmoidFunction(1f), 
                    },
                        new int[]
                    {
                        784,
                        500,
                        10
                    });


                using (var clProvider = new CLProvider())
                {
                    var config = new LearningAlgorithmConfig(
                        new ConstLearningRate(0.02f),
                        1,
                        0.0f,
                        1,
                        0.0001f,
                        -1.0f);

                    var validation = new ClassificationValidation(
                        new ConsoleMLPSaver(), 
                        new HalfSquaredEuclidianDistance(),
                        validationData,
                        300,
                        100);

                    var alg =
                        new BackpropagationAlgorithm(
                            randomizer,
                            new DropConnectCPUBackpropagationEpocheTrainer<VectorizedCPULayerInferenceV2>(
                                randomizer,
                                VectorizationSizeEnum.VectorizationMode16,
                                mlp,
                                config,
                                clProvider,
                                sampleCount,
                                p),
                            mlp,
                            validation,
                            config);

                    //обучение сети
                    alg.Train(
                        new NoDeformationTrainDataProvider(trainData));
                }
            }
        }
    }
}
