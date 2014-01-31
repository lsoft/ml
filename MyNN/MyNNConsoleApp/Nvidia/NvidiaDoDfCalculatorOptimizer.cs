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
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Half;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.Nvidia
{
    public class NvidiaDoDfCalculatorOptimizer
    {
        public static void Optimize()
        {
            //const int DataItemCount = 10001;//10001;
            //const int DataItemLength = 787;//787;

            //int genSeed = DateTime.Now.Millisecond;
            //var genRandomizer = new DefaultRandomizer(ref genSeed);

            //var diList = new List<DataItem>();
            //for (var cc = 0; cc < DataItemCount; cc++)
            //{
            //    var i = new float[DataItemLength];
            //    var o = new float[1];

            //    for (var dd = 0; dd < DataItemLength; dd++)
            //    {
            //        i[dd] =
            //            //((dd%2) > 0) ? 1f : 0f;
            //            //dd / (float)DataItemLength + cc;
            //            //dd*0.015625f + cc;
            //            //genRandomizer.Next(10000) * 0.01f;
            //            //genRandomizer.Next(10000)*0.015625f;
            //            genRandomizer.Next(100);

            //    }

            //    var di = new DataItem(i, o);
            //    diList.Add(di);
            //}

            //var dataset = new DataSet(diList);

            var dataset = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1000
                );
            dataset.Normalize();

            DodfDictionary nvidiaResult;
            {
                var randomizer = new NoRandomRandomizer();

                nvidiaResult = ProfileNvidiaGPU(
                    randomizer,
                    dataset);
            }

            DodfDictionary intelResult;
            {
                var randomizer = new NoRandomRandomizer();

                intelResult = ProfileIntelCPU(
                    randomizer,
                    dataset);
            }


            if (nvidiaResult.Count != intelResult.Count)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("Intel sizes != Nvidia sizes");
                return;
            }

            float maxDiff = float.MinValue;
            for (var cc = 0; cc < nvidiaResult.Count; cc++)
            {
                for (var dd = cc; dd < nvidiaResult.Count; dd++)
                {
                    var diff = Math.Abs(nvidiaResult.GetDistance(cc, dd) - intelResult.GetDistance(cc, dd));

                    maxDiff = Math.Max(maxDiff, diff);
                }

                Console.WriteLine();
            }

            Console.ForegroundColor = Math.Abs(maxDiff) > 1e-7 ? ConsoleColor.Red : ConsoleColor.Green;
            Console.WriteLine("Finished with MAXDIFF = {0}", maxDiff);
            Console.ResetColor();
        }

        private static DodfDictionary ProfileNvidiaGPU(
            NoRandomRandomizer randomizer,
            DataSet dataset)
        {
            //var dd = new GPUNaiveDistanceDictFactory(
            //    new NvidiaOrAmdGPUDeviceChooser());
            var dd = new GPUHalfNaiveDistanceDictFactory(
                new NvidiaOrAmdGPUDeviceChooser());

            TimeSpan takenTime;
            var result = dd.CreateDistanceDict(dataset.Data, out takenTime);

            Console.WriteLine(
                "NVIDIA TAKES {0}",
                takenTime);

            return result;
        }

        private static DodfDictionary ProfileIntelCPU(
            NoRandomRandomizer randomizer,
            DataSet dataset)
        {
            var dd = new VectorizedCPUDistanceDictFactory();

            TimeSpan takenTime;
            var result = dd.CreateDistanceDict(dataset.Data, out takenTime);

            Console.WriteLine(
                "INTEL  TAKES {0}",
                takenTime);

            return result;
        }
    }
}
