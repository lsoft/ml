using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Common;
using System.Linq;
using System.Text;
using System.Threading;
using MyNN.Data;
using MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator;
using MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator.OpenCL;
using MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator.OpenCL.DistanceDict;

namespace MyNNConsoleApp
{
    public class pabProfiler
    {
        public static void Main2()
        {
            var data = Generate();

            var beforeCreate0 = DateTime.Now;
            var pabOld = new DodfCalculatorVectorized(data);
            var afterCreate0 = DateTime.Now;

            var beforeCreate1 = DateTime.Now;
            var pabNew = new DodfCalculatorOpenCL(
                data,
                new OpenCLDistanceDictFactory());
            var afterCreate1 = DateTime.Now;

            Console.WriteLine(
                "OLD TAKES {0}",
                (afterCreate0 - beforeCreate0));
            Console.WriteLine(
                "NEW TAKES {0}",
                (afterCreate1 - beforeCreate1));

            for (var dd = 0; dd < data.Count; dd++)
            {
                var oldItem = pabOld.CalculateDodf(dd);
                var newItem = pabNew.CalculateDodf(dd);

                for (var cc = 0; cc < oldItem.Length; cc++)
                {
                    var diff = oldItem[cc] - newItem[cc];

                    if (Math.Abs(diff) > 1e-5)
                    {
                        throw new InvalidOperationException("Math.Abs(diff) > 1e-5");
                    }
                }

                if (dd%100 == 0)
                {
                    Console.WriteLine("{0} out of {1}", dd, data.Count);
                }
            }

            Console.WriteLine("SUCCESS");
        }

        private static List<DataItem> Generate()
        {
            const int count =
                //5;
                3000;
                //12000;
            const int inputLength =
                50;
                //100;
            const int classesCount = 10;

            var rnd = new Random(DateTime.Now.Millisecond);

            var result = new List<DataItem>();

            for (var cc = 0; cc < count; cc++)
            {
                var input = new float[inputLength];
                var output = new float[classesCount];

                for (var dd = 0; dd < inputLength; dd++)
                {
                    input[dd] = (float) rnd.NextDouble();
                }

                output[rnd.Next(classesCount)] = 1f;

                result.Add(new DataItem(input, output));
            }

            return result;
        }
    }
}
