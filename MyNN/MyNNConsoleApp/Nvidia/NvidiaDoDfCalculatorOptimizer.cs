using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.OpenCL.DeviceChooser;

namespace MyNNConsoleApp.Nvidia
{
    public class NvidiaDoDfCalculatorOptimizer
    {
        public static void Optimize()
        {
            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                //int.MaxValue
                1000
                );
            trainData.Data.ForEach(item => item.Input.Transform((value) => value + 0.1f)); //чтобы не было нулей в датасете, а то вдруг алгоритм "забывает" например учесть последний флоат в датаитеме...
            //trainData.Normalize();
            //trainData = new DataSet(
            //    trainData.Take(16).ToList(),
            //    trainData.Visualizer);

            Func<ILearningAlgorithmConfig> configProvider =
                () =>
                    new LearningAlgorithmConfig(
                        new ConstLearningRate(0.0001f),
                        1,
                        0.0f,
                        1,
                        0.0001f,
                        -1.0f);

            Dictionary<int, float[]> nvidiaResult;
            {
                var randomizer = new NoRandomRandomizer();

                nvidiaResult = ProfileNvidiaGPU(
                    randomizer,
                    trainData);
            }

            Dictionary<int, float[]> intelResult;
            {
                var randomizer = new NoRandomRandomizer();

                intelResult = ProfileIntelCPU(
                    randomizer,
                    trainData);
            }


            if (intelResult == null || nvidiaResult == null)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Fail to obtain results!");
                return;
            }

            if (intelResult.Count != nvidiaResult.Count)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Intel size {0} != Nvidia size {1}", intelResult.Count, nvidiaResult.Count);
                return;
            }

            var intelkeys = intelResult.Keys.ToArray();
            var nvidiakeys = nvidiaResult.Keys.ToArray();

            if (!ArrayOperations.ValuesAreEqual(intelkeys, nvidiakeys))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Intel keys != Nvidia keys");
                return;
            }

            var maxDiff = float.MinValue;
            for (var index = 0; index < intelResult.Count; index++)
            {
                var key = intelResult.Keys.ToArray()[index];

                var intelValues = intelResult[key];
                var nvidiaValues = nvidiaResult[key];

                float diff;
                if (!ArrayOperations.ValuesAreEqual(intelValues, nvidiaValues, 1e-7f, out diff))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Intel value != Nvidia value with DIFF = {0}", diff);
                    return;
                }

                if (diff > maxDiff)
                {
                    maxDiff = diff;
                }
            }

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("OK with MAXDIFF = {0}", maxDiff);

        }

        private static Dictionary<int, float[]> ProfileNvidiaGPU(
            NoRandomRandomizer randomizer,
            DataSet trainData)
        {
            var dd = new GPUNaiveDistanceDictFactory(
                new NvidiaOrAmdGPUDeviceChooser());

            TimeSpan takenTime;
            var result = dd.CreateDistanceDict(trainData.Data, out takenTime);

            Console.WriteLine(
                "NVIDIA TAKES {0}",
                takenTime);

            return result;
        }

        private static Dictionary<int, float[]> ProfileIntelCPU(
            NoRandomRandomizer randomizer,
            DataSet trainData)
        {
            var dd = new VOpenCLDistanceDictFactory();

            TimeSpan takenTime;
            var result = dd.CreateDistanceDict(trainData.Data, out takenTime);

            Console.WriteLine(
                "INTEL  TAKES {0}",
                takenTime);

            return result;
        }
    }
}
