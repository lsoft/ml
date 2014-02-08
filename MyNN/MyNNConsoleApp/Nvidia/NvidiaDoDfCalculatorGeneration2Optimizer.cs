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
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation3;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation3.Float;
using MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation3.Half;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPU;
using MyNN.MLP2.LearningConfig;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using DefaultRandomizer = MyNN.Randomizer.DefaultRandomizer;

namespace MyNNConsoleApp.Nvidia
{
    public class NvidiaDoDfCalculatorGeneration2Optimizer
    {
        public static void Optimize()
        {
            float maxDiff = float.MinValue;

            //const int DataItemCount = 10001;
            //const int DataItemLength = 787;

            //int genSeed = 123; //DateTime.Now.Millisecond;
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
                //int.MaxValue
                1000
                );
            dataset.Normalize();


            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            //////////int seed = 1;
            //////////var mlp = new MLP(
            //////////    new DefaultRandomizer(ref seed),
            //////////    null,
            //////////    null,
            //////////    new IFunction[]
            //////////    {
            //////////        null,
            //////////        new HyperbolicTangensFunction(), 
            //////////    },
            //////////    new int[]
            //////////    {
            //////////        28*28,
            //////////        500
            //////////    });

            //////////using (var cl = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(), true))
            //////////{
            //////////    var forward = new GPUForwardPropagation(
            //////////        mlp,
            //////////        cl);

            //////////    var output = forward.ComputeOutput(dataset);

            //////////    var dd = new GpuHalfDistanceDictCalculator(
            //////////        new NvidiaOrAmdGPUDeviceChooser());

            //////////    TimeSpan takenTime;

            //////////    var result = dd.CreateDistanceDict(
            //////////        output.ConvertAll(j => new DataItem(j.State, new float[1])),
            //////////        out takenTime);

            //////////    Console.WriteLine(
            //////////        "TEST TAKES {0}",
            //////////        takenTime);
            //////////}

            //////////return;

            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



            dataset.Data.ForEach(j => j.Input.Transform((a) => ((int)(a  * 256)) / 256f));

            for (var iteration = 0; iteration < 1; iteration++)
            {
                ConsoleAmbientContext.Console.WriteLine("\r\n----------------------- Iteration {0} --------------------------", iteration);

                DodfDistanceContainer nvidiaResult;
                {
                    //nvidiaResult = null;
                    nvidiaResult = ProfileNvidiaGPU(
                        dataset);
                }

                DodfDistanceContainer intelResult;
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

                DumpDict(
                    "NVIDIA",
                    nvidiaResult);

                DumpDict(
                    "INTEL",
                    intelResult);

                //if (nvidiaResult.Count != intelResult.Count || nvidiaResult.Length != intelResult.Length)
                //{
                //    Console.ForegroundColor = ConsoleColor.Red;
                //    Console.WriteLine("Intel sizes != Nvidia sizes");
                //    return;
                //}

                maxDiff = float.MinValue;
                for (var cc = 0; cc < nvidiaResult.Count; cc++)
                {
                    for (var dd = cc; dd < nvidiaResult.Count; dd++)
                    {
                        var diff = Math.Abs(nvidiaResult.GetDistance(cc, dd) - intelResult.GetDistance(cc, dd));

                        maxDiff = Math.Max(maxDiff, diff);
                    }
                }

                if (maxDiff > 0f)
                {
                    break;
                }
            }

            Console.ForegroundColor = Math.Abs(maxDiff) > 1e-7 ? ConsoleColor.Red : ConsoleColor.Green;
            Console.WriteLine("Finished with MAXDIFF = {0}", maxDiff);
            Console.ResetColor();
        }

        private static void DumpDict(
            string vendor,
            DodfDistanceContainer results)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            
            Console.WriteLine(vendor);

            if (results.Count <= 10)
            {
                for (var cc = 0; cc < results.Count; cc++)
                {
                    Console.Write("{0:D7}: ", cc);

                    for (var dd = cc; dd < results.Count; dd++)
                    {
                        var distance = results.GetDistance(cc, dd);

                        Console.SetCursorPosition(dd * 11 + 10, Console.CursorTop);

                        if (distance >= 0)
                        {
                            Console.Write(" ");
                        }

                        Console.Write("{0:000000.00}", distance);
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

        private static DodfDistanceContainer ProfileNvidiaGPU(
            DataSet dataset)
        {
            var dd = new GpuHalfDistanceDictCalculator(
                new NvidiaOrAmdGPUDeviceChooser());

            TimeSpan takenTime;
            var result = dd.CreateDistanceDict(dataset.Data, out takenTime);

            Console.WriteLine(
                "NVIDIA TAKES {0}",
                takenTime);

            return result;
        }

        private static DodfDistanceContainer ProfileIntelCPU(
            DataSet dataset)
        {
            var dd = new VectorizedCpuDistanceDictCalculator();

            TimeSpan takenTime;
            var result = dd.CreateDistanceDict(dataset.Data, out takenTime);

            Console.WriteLine(
                "INTEL  TAKES {0}",
                takenTime);

            return result;
        }
    }
}
