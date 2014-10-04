using System;
using System.CodeDom;
using System.Windows.Markup;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

using InferenceAlias = MyNN.MLP2.ForwardPropagation.DropConnect.Inference;

namespace MyNN.Tests.MLP2.Forward.DropConnect.Inferencer
{
    [TestClass]
    public class InferenceFixture
    {
        private const float Epsilon = 1e-5f;

        [TestMethod]
        public void CSharpNaiveLayerInferencerTest()
        {
            var tester = new InferencerTester<InferenceAlias.CSharp.NaiveLayerInference>(
                );

            float[] orig;
            float[] test;
            tester.Test(
                new IntelCPUDeviceChooser(true), 
                new LinearFunction(1f),
                1000000,
                1f,
                out orig,
                out test
                );

            for (var cc = 0; cc < orig.Length; cc++)
            {
                if (float.IsNaN(orig[cc]) || float.IsNaN(test[cc]))
                {
                    Assert.Fail("NaN");
                }

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

        [TestMethod]
        public void OpenCLCPUNaiveLayerInferencerTest()
        {
            var tester = new InferencerTester<InferenceAlias.OpenCL.CPU.NaiveLayerInference>(
                );

            float[] orig;
            float[] test;
            tester.Test(
                new IntelCPUDeviceChooser(true),
                new LinearFunction(1f),
                1000000,
                1f,
                out orig,
                out test
                );

            for (var cc = 0; cc < orig.Length; cc++)
            {
                if (float.IsNaN(orig[cc]) || float.IsNaN(test[cc]))
                {
                    Assert.Fail("NaN");
                }

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

        [TestMethod]
        public void OpenCLCPUDefaultLayerInferencerTest()
        {
            var tester = new InferencerTester<InferenceAlias.OpenCL.CPU.DefaultLayerInference>(
                );

            float[] orig;
            float[] test;
            tester.Test(
                new IntelCPUDeviceChooser(true),
                new LinearFunction(1f),
                1000000,
                1f,
                out orig,
                out test
                );

            for (var cc = 0; cc < orig.Length; cc++)
            {
                if (float.IsNaN(orig[cc]) || float.IsNaN(test[cc]))
                {
                    Assert.Fail("NaN");
                }

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

        [TestMethod]
        public void OpenCLCPUVectorizedLayerInferencerTest()
        {
            var tester = new InferencerTester<InferenceAlias.OpenCL.CPU.VectorizedLayerInference>(
                );

            float[] orig;
            float[] test;
            tester.Test(
                new IntelCPUDeviceChooser(true),
                new LinearFunction(1f),
                1000000,
                1f,
                out orig,
                out test
                );

            for (var cc = 0; cc < orig.Length; cc++)
            {
                if (float.IsNaN(orig[cc]) || float.IsNaN(test[cc]))
                {
                    Assert.Fail("NaN");
                }

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


        [TestMethod]
        public void GPULayerInferencerTest()
        {
            var tester = new InferencerTester<InferenceAlias.OpenCL.GPU.GPULayerInference>(
                );

            float[] orig;
            float[] test;
            tester.Test(
                new NvidiaOrAmdGPUDeviceChooser(true), 
                new LinearFunction(1f),
                1000000,
                1f,
                out orig,
                out test
                );

            for (var cc = 0; cc < orig.Length; cc++)
            {
                if (float.IsNaN(orig[cc]) || float.IsNaN(test[cc]))
                {
                    Assert.Fail("NaN");
                }

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
