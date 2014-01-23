using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL;
using MyNN.MLP2.Backpropagaion.Metrics;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.DeviceChooser;

namespace MyNNConsoleApp.Nvidia
{
    public class NvidiaBackpropOptimizer
    {
        public static void Optimize()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                100
                );
            trainData.Normalize();
            trainData = new DataSet(
                new List<DataItem>
                {
                    trainData[0]
                    ,trainData[1]
                    ,trainData[2]
                    ,trainData[3]
                    ,trainData[4]
                    ,trainData[5]
                    ,trainData[6]
                    ,trainData[7]
                    ,trainData[8]
                    ,trainData[9]
                },
                trainData.Visualizer);
            trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                1
                );
            validationData.Normalize();

            var config = new LearningAlgorithmConfig(
                new ConstLearningRate(0.1f),
                1,
                0.0f,
                1,
                0.0001f,
                -1.0f);

            var serialization = new SerializationHelper();

            var validation = new AutoencoderValidation(
                serialization,
                new HalfSquaredEuclidianDistance(),
                validationData.ConvertToAutoencoder(),
                3,
                1);

            var mlp = new MLP(
                new NoRandomRandomizer(), 
                null,
                null,
                new IFunction[]
                    {
                        null,
                        new SigmoidFunction(1f),
                        new SigmoidFunction(1f),
                    },
                new[]
                    {
                        784,
                        784 * 10,
                        784
                    });


            //{
            //    var randomizer = new NoRandomRandomizer();

            //    var config2 = new LearningAlgorithmConfig(
            //        new ConstLearningRate(0.1f),
            //        1,
            //        0.0f,
            //        10,
            //        0.0001f,
            //        -1.0f);

            //    ProfileNvidiaGPU(
            //        randomizer,
            //        trainData,
            //        serialization.DeepClone(mlp),
            //        config2,
            //        validation);
            //}

            //Console.Clear();


            {
                var randomizer = new NoRandomRandomizer();

                ProfileNvidiaGPU(
                    randomizer,
                    trainData,
                    serialization.DeepClone(mlp),
                    config,
                    validation);
            }

            {
                var randomizer = new NoRandomRandomizer();

                ProfileIntelCPU(
                    randomizer,
                    trainData,
                    serialization.DeepClone(mlp),
                    config,
                    validation);
            }
        }

        private static void ProfileNvidiaGPU(
            IRandomizer randomizer,
            DataSet trainData,
            MLP mlp,
            ILearningAlgorithmConfig config,
            IValidation validation)
        {
            using (var clProvider = new CLProvider(
                new NvidiaOrAmdGPUDeviceChooser(),
                true))
            {
                var alg =
                    new BackpropagationAlgorithm(
                        randomizer,
                        (currentMLP, currentConfig) =>
                            new GPUBackpropagationAlgorithm(
                                currentMLP,
                                currentConfig,
                                clProvider),
                        mlp,
                        validation,
                        config);

                //обучение сети
                alg.Train(
                    new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
            }
        }

        private static void ProfileIntelCPU(
            IRandomizer randomizer,
            DataSet trainData,
            MLP mlp,
            ILearningAlgorithmConfig config,
            IValidation validation)
        {
            using (var clProvider = new CLProvider(
                new IntelCPUDeviceChooser(),
                true))
            {

                var alg =
                    new BackpropagationAlgorithm(
                        randomizer,
                        (currentMLP, currentConfig) =>
                            new OpenCLBackpropagationAlgorithm(
                                VectorizationSizeEnum.VectorizationMode16,
                                currentMLP,
                                currentConfig,
                                clProvider),
                        mlp,
                        validation,
                        config);

                //обучение сети
                alg.Train(
                    new NoDeformationTrainDataProvider(trainData).GetDeformationDataSet);
            }
        }

    }
}
