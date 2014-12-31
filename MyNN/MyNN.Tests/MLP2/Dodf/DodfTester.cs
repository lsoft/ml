using System;
using System.Collections.Generic;
using System.Diagnostics;
using MyNN.Common.NewData.Item;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.CSharp;

namespace MyNN.Tests.MLP2.Dodf
{
    internal class DodfTester
    {
        private const float Epsilon = 1e-5f;

        public static bool DoTest(
            Func<List<IDataItem>, IDodfCalculator> dodfCalculatorFunc
            )
        {
            if (dodfCalculatorFunc == null)
            {
                throw new ArgumentNullException("dodfCalculatorFunc");
            }

            const int ItemCount = 100;
            const int InputSize = 100;
            const int OutputSize = 2;
            var rnd = new Random(123);

            var dil = new List<IDataItem>();
            for (var cc = 0; cc < ItemCount; cc++)
            {
                var dinput = new float[InputSize];
                dinput.Fill(j => (float)rnd.NextDouble() );

                var doutput = new float[OutputSize];
                doutput[rnd.Next(OutputSize)] = 1f;

                var di = new DataItem(
                    dinput,
                    doutput
                    );

                dil.Add(di);
            }

            var dodfc0 = new DodfCalculatorOld(
                dil
                );


            var dodfc1 = dodfCalculatorFunc(
                dil
                );

            for (var cc = 0; cc < ItemCount; cc++)
            {
                var dodf0 = dodfc0.CalculateDodf(
                    cc
                    );

                var dodf1 = dodfc1.CalculateDodf(
                    cc
                    );

                float maxDiff;
                var eq = ArrayOperations.ValuesAreEqual(
                    dodf0,
                    dodf1,
                    Epsilon,
                    out maxDiff
                    );

                if (!eq)
                {
                    ConsoleAmbientContext.Console.WriteLine("dodf0:");
                    ConsoleAmbientContext.Console.WriteLine(
                        string.Join(
                            " ",
                            dodf0));

                    ConsoleAmbientContext.Console.WriteLine("dodf1:");
                    ConsoleAmbientContext.Console.WriteLine(
                        string.Join(
                            " ",
                            dodf1));

                    ConsoleAmbientContext.Console.WriteLine(
                        "Max diff: {0}",
                        maxDiff
                        );

                    return false;
                }
            }

            return true;
        }
    }
}