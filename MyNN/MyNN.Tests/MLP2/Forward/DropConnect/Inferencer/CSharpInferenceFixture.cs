using System;
using System.CodeDom;
using System.Windows.Markup;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.Forward.DropConnect.Inferencer
{
    [TestClass]
    public class CSharpInferenceFixture
    {
        private const float Epsilon = 1e-6f;

        [TestMethod]
        public void TestMethod1()
        {
            var tester = new InferencerTester(
                );

            float[] orig;
            float[] test;
            tester.Test(
                new LinearFunction(1f),
                1000000,
                1f,
                out orig,
                out test
                );

            for (var cc = 0; cc < orig.Length; cc++)
            {
                var diff = Math.Abs(orig[cc] - test[cc]);

                if (diff >= Epsilon)
                {
                    ConsoleAmbientContext.Console.WriteLine(
                        string.Format(
                            "Orig = {0}, test = {1}, diff = {2}, at index = {3}",
                            orig[cc],
                            test[cc],
                            diff,
                            cc
                            ));

                    Dump("Original:", orig);
                    Dump("Test:", test);

                    Assert.Fail("diff >= Epsilon");
                }

            }

            ConsoleAmbientContext.Console.WriteLine("Success!");

            Dump("Original:", orig);
            Dump("Test:", test);


            

        }


        private void Dump(string message, float[] a)
        {
            if (message == null)
            {
                throw new ArgumentNullException("message");
            }
            if (a == null)
            {
                throw new ArgumentNullException("a");
            }

            ConsoleAmbientContext.Console.WriteLine(message);

            foreach (var i in a)
            {
                ConsoleAmbientContext.Console.Write(
                    string.Format(
                        "{0};",
                        i));
            }
            ConsoleAmbientContext.Console.WriteLine(string.Empty);
        }

    }
}
