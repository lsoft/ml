using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Common;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using MyNN.Data;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict;
using MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation2;
using OpenCL.Net.OpenCL.DeviceChooser;

namespace MyNNConsoleApp
{
    public class pabProfiler
    {
        public static void Main2()
        {
            var data = Generate();

            var beforeCreate0 = DateTime.Now;
            //var dodf0 = new DodfCalculatorVectorized(data);
            var dodf0 = new DodfCalculatorOld(data);
            var afterCreate0 = DateTime.Now;

            var beforeCreate1 = DateTime.Now;
            var dodf1 = new DodfCalculatorOpenCL(
                data,
                //new VOpenCLDistanceDictFactory());
                new GPUHalfDistanceDictFactory(new NvidiaOrAmdGPUDeviceChooser()));
            var afterCreate1 = DateTime.Now;

            Console.WriteLine(
                "C# TAKES {0}",
                (afterCreate0 - beforeCreate0));
            Console.WriteLine(
                "CPU OPENCL TAKES {0}",
                (afterCreate1 - beforeCreate1));

            int index = 0;

            var beforeChecking = DateTime.Now;

            //Parallel.For(0, data.Count, dd =>
            for(var dd = 0; dd < data.Count; dd++)
            {
                var oldItem = dodf0.CalculateDodf(dd);
                var newItem = dodf1.CalculateDodf(dd);

                for (var cc = 0; cc < oldItem.Length; cc++)
                {
                    var diff = oldItem[cc] - newItem[cc];

                    if (Math.Abs(diff) > 1e-15)
                    {
                        throw new InvalidOperationException(
                            string.Format(
                                "diff = {0} > 1e-15, left= {1}, right = {2}",
                                Math.Abs(diff),
                                oldItem[cc],
                                newItem[cc]));
                    }
                }

                var newIndex = Interlocked.Increment(ref index);
                if (newIndex  % 100 == 0)
                {
                    Console.WriteLine("{0} out of {1}", newIndex, data.Count);
                }
            }
            //);//Parallel.For

            var afterChecking = DateTime.Now;

            Console.WriteLine(
                "CHECKING TAKES {0}",
                (afterChecking - beforeChecking));

            Console.WriteLine("SUCCESS");
        }

        private static List<DataItem> Generate()
        {
            const int count =
                //5;
                150;
                //1000;
                //3000;
                //12000;
            const int inputLength =
                50;
                //100;
                //200;
            const int classesCount = 10;

            var rnd = new Random(DateTime.Now.Millisecond);

            var result = new List<DataItem>();

            for (var cc = 0; cc < count; cc++)
            {
                var input = new float[inputLength];
                var output = new float[classesCount];

                for (var dd = 0; dd < inputLength; dd++)
                {
                    input[dd] =
                        rnd.Next(1000) * 0.015625f;
                        //(float) rnd.NextDouble();
                }

                output[rnd.Next(classesCount)] = 1f;

                result.Add(new DataItem(input, output));
            }

            return result;
        }
    }
}
