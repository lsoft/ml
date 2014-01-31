﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.CPU.Default;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.OpenCL.GPU.Default;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.Nvidia
{
    public class NvidiaBackpropOptimizer
    {
        public static void Optimize()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                //int.MaxValue
                100
                );
            trainData.Data.ForEach(item => item.Input.Transform((value) => value + 0.1f)); //чтобы не было нулей в датасете, а то вдруг алгоритм "забывает" например учесть последний флоат в датаитеме...
            //trainData.Normalize();
            //trainData = new DataSet(
            //    trainData.Take(50).ToList(),
            //    trainData.Visualizer);
            trainData = trainData.ConvertToAutoencoder();

            var validationData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/testset/",
                //int.MaxValue
                100
                );
            validationData.Normalize();

            Func<ILearningAlgorithmConfig> configProvider = 
                () =>
                    new LearningAlgorithmConfig(
                        new ConstLearningRate(0.0001f),
                        1,
                        0.0f,
                        1,
                        0.0001f,
                        -1.0f);

            var serialization = new SerializationHelper();

            Func<TestPurposeValidation> validationProvider = 
                () =>
                    new TestPurposeValidation(validationData.ConvertToAutoencoder());

            Func<MLP> mlpProvider = 
                () =>
                    new MLP(
                        new NoRandomRandomizer(), 
                        null,
                        null,
                        new IFunction[]
                            {
                                null,
                                new RLUFunction(), 
                                new RLUFunction(),
                            },
                        new[]
                            {
                                784,
                                1200,//784 * 10,
                                784
                            });

            var nvidiaTotalError = float.MinValue;
            var nvidiaResult = ulong.MinValue;
            {
                var randomizer = new NoRandomRandomizer();
                var validation = validationProvider();
                var mlp = mlpProvider();
                var config = configProvider();

                ProfileNvidiaGPU(
                    randomizer,
                    trainData,
                    mlp,
                    config,
                    validation);

                nvidiaTotalError = validation.TotalError;
                nvidiaResult = validation.TotalSum;
            }

            var intelTotalError = float.MaxValue;
            var intelResult = ulong.MaxValue;
            {
                var randomizer = new NoRandomRandomizer();
                var validation = validationProvider();
                var mlp = mlpProvider();
                var config = configProvider();

                ProfileIntelCPU(
                    randomizer,
                    trainData,
                    mlp,
                    config,
                    validation);

                intelTotalError = validation.TotalError;
                intelResult = validation.TotalSum;
            }

            var diff = nvidiaTotalError >= intelTotalError ? nvidiaTotalError - intelTotalError : intelTotalError - nvidiaTotalError;
            var diff2 = BitConverter.GetBytes(diff);
            Console.WriteLine("diff double\r\n{4}\r\ndiff bytes\r\n{0} {1} {2} {3}\r\n", diff2[0], diff2[1], diff2[2], diff2[3], DoubleConverter.ToExactString(diff));

            var nvidiaTotalError2 = BitConverter.GetBytes(nvidiaTotalError);
            var intelTotalError2 = BitConverter.GetBytes(intelTotalError);

            var nvidiaTotalError3 = BitConverter.ToUInt32(nvidiaTotalError2, 0);
            var intelTotalError3 = BitConverter.ToUInt32(intelTotalError2, 0);

            var nvidiaTotalError4 = DoubleConverter.ToExactString(nvidiaTotalError);
            var intelTotalError4 = DoubleConverter.ToExactString(intelTotalError);

            Console.WriteLine("nvidia bytes\r\n{0} {1} {2} {3}\r\n", nvidiaTotalError2[0], nvidiaTotalError2[1], nvidiaTotalError2[2], nvidiaTotalError2[3]);
            Console.WriteLine("intel  bytes\r\n{0} {1} {2} {3}\r\n", intelTotalError2[0], intelTotalError2[1], intelTotalError2[2], intelTotalError2[3]);
            Console.WriteLine("uint\r\n{0}\r\n{1}\r\n", nvidiaTotalError3, intelTotalError3);
            Console.WriteLine("double\r\n{0}\r\n{1}\r\n", nvidiaTotalError4, intelTotalError4);

            if(Math.Abs(nvidiaTotalError - intelTotalError) < float.Epsilon)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("RESULTS ARE EQUALS");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("RESULTS ARE NOT EQUALS \r\n{0} = nvidia\r\n{1} = intel", DoubleConverter.ToExactString(nvidiaTotalError), DoubleConverter.ToExactString(intelTotalError));
                Console.ResetColor();
            }

            //if 
            //    //(Math.Abs(nvidiaTotalError - intelTotalError) < float.Epsilon)
            //    (nvidiaResult == intelResult)
            //{
            //    Console.ForegroundColor = ConsoleColor.Green;
            //    Console.WriteLine("RESULTS ARE EQUALS ({0})", nvidiaResult);
            //    Console.ResetColor();
            //}
            //else if (((nvidiaResult >= intelResult) ? (nvidiaResult - intelResult) : (intelResult - nvidiaResult)) < 1000)
            //{
            //    Console.ForegroundColor = ConsoleColor.Yellow;
            //    Console.WriteLine("RESULTS ARE NOT EQUALS \r\n{0} = nvidia\r\n{1} = intel", nvidiaResult, intelResult);
            //    Console.ResetColor();
            //}

            //else
            //{
            //    Console.ForegroundColor = ConsoleColor.Red;
            //    Console.WriteLine("RESULTS ARE NOT EQUALS \r\n{0} = nvidia\r\n{1} = intel", nvidiaResult, intelResult);
            //    Console.ResetColor();
            //}
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
                            new CPUBackpropagationAlgorithm(
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
