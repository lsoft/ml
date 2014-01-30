using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using MyNN;
using MyNN.Data;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation2;
using MyNN.MLP2.Backpropagaion.Validation;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.OpenCL.DeviceChooser;

namespace MyNNConsoleApp.Nvidia
{
    public class NvidiaDoDfCalculatorGeneration2Optimizer
    {
        public static void Optimize()
        {
            const int DataItemCount = 10001;//10001;
            const int DataItemLength = 787;//787;

            int genSeed = DateTime.Now.Millisecond;
            var genRandomizer = new DefaultRandomizer(ref genSeed);

            var diList = new List<DataItem>();
            for (var cc = 0; cc < DataItemCount; cc++)
            {
                var i = new float[DataItemLength];
                var o = new float[1];

                for (var dd = 0; dd < DataItemLength; dd++)
                {
                    i[dd] =
                        //((dd%2) > 0) ? 1f : 0f;
                        //dd / (float)DataItemLength + cc;
                        //dd*0.015625f + cc;
                        //genRandomizer.Next(10000) * 0.01f;
                        //genRandomizer.Next(10000)*0.015625f;
                        genRandomizer.Next(100);

                }

                var di = new DataItem(i, o);
                diList.Add(di);
            }

            var dataset = new DataSet(diList);

            //var dataset = MNISTDataProvider.GetDataSet(
            //    "_MNIST_DATABASE/mnist/trainingset/",
            //    2000
            //    );
            //dataset.Normalize();

            Dictionary<int, float[]> nvidiaResult;
            {
                nvidiaResult = ProfileNvidiaGPU(
                    dataset);
            }

            Dictionary<int, float[]> intelResult;
            {
                intelResult = ProfileIntelCPU(
                    dataset);
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

            DumpDict(
                "NVIDIA",
                nvidiaResult);

            DumpDict(
                "INTEL",
                intelResult);

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
            Console.ResetColor();
        }

        private static void DumpDict(
            string vendor,
            Dictionary<int, float[]> results)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            
            Console.WriteLine(vendor);

            var maxLength = results.Max(j => j.Value.Length);


            if (results.Count <= 10 && maxLength <= 10)
            {

                foreach (var pair in results)
                {
                    Console.Write("{0:D7}: ", pair.Key);

                    for (var cc = 0; cc < maxLength - pair.Value.Length; cc++)
                    {
                        Console.Write("          ");
                    }

                    foreach (var v in pair.Value)
                    {
                        Console.Write("{0:000000.00} ", v);
                    }

                    Console.WriteLine();
                }
            }
            else
            {
                Console.WriteLine("Too big size to diplay.");
            }

            Console.ResetColor();
        }

        private static Dictionary<int, float[]> ProfileNvidiaGPU(
            DataSet dataset)
        {
            var dd = new GPUHalfDistanceDictFactory(
                new NvidiaOrAmdGPUDeviceChooser());

            TimeSpan takenTime;
            var result = dd.CreateDistanceDict(dataset.Data, out takenTime);

            Console.WriteLine(
                "NVIDIA TAKES {0}",
                takenTime);

            return result;
        }

        private static Dictionary<int, float[]> ProfileIntelCPU(
            DataSet dataset)
        {
            var dd = new VOpenCLDistanceDictFactory();

            TimeSpan takenTime;
            var result = dd.CreateDistanceDict(dataset.Data, out takenTime);

            Console.WriteLine(
                "INTEL  TAKES {0}",
                takenTime);

            return result;
        }
    }
}
