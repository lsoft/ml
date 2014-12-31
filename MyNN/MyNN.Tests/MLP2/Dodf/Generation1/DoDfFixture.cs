using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.CSharp;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL;
using MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation1;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.Tests.MLP2.Dodf.Generation1
{
    [TestClass]
    public class DoDfFixture
    {
        [TestInitialize()]
        public void Initialize()
        {
            DoDfAmbientContext.DisableExponential = true;
        }

        [TestCleanup()]
        public void Cleanup()
        {
            DoDfAmbientContext.DisableExponential = false;
        }

        [TestMethod]
        public void DodfCalculatorTest()
        {
            var eq = DodfTester.DoTest(
                (dil) =>
                {
                    return
                        new DodfCalculator(
                            dil
                            );
                });

            Assert.IsTrue(eq);
        }

        [TestMethod]
        public void DodfCalculatorVectorizedTest()
        {
            var eq = DodfTester.DoTest(
                (dil) =>
                {
                    return
                        new DodfCalculatorVectorized(
                            dil
                            );
                });

            Assert.IsTrue(eq);
        }

        [TestMethod]
        public void DodfCalculatorVectorizedDebugTest()
        {
            var eq = DodfTester.DoTest(
                (dil) =>
                {
                    return
                        new DodfCalculatorVectorizedDebug(
                            dil
                            );
                });

            Assert.IsTrue(eq);
        }

        [TestMethod]
        public void DodfCalculatorOpenCLTest0()
        {
            var eq = DodfTester.DoTest(
                (dil) =>
                {
                    return
                        new DodfCalculatorOpenCL(
                            dil,
                            new CSharpDistanceDictCalculator()
                            );
                });

            Assert.IsTrue(eq);
        }

        [TestMethod]
        public void DodfCalculatorOpenCLTest1()
        {
            var eq = DodfTester.DoTest(
                (dil) =>
                {
                    return
                        new DodfCalculatorOpenCL(
                            dil,
                            new CpuDistanceDictCalculator()
                            );
                });

            Assert.IsTrue(eq);
        }

        [TestMethod]
        public void DodfCalculatorOpenCLTest2()
        {
            var eq = DodfTester.DoTest(
                (dil) =>
                {
                    return
                        new DodfCalculatorOpenCL(
                            dil,
                            new VectorizedCpuDistanceDictCalculator()
                            );
                });

            Assert.IsTrue(eq);
        }

        [TestMethod]
        public void DodfCalculatorOpenCLTest3()
        {
            var eq = DodfTester.DoTest(
                (dil) =>
                {
                    return
                        new DodfCalculatorOpenCL(
                            dil,
                            new GpuNaiveDistanceDictCalculator(
                                new NvidiaOrAmdGPUDeviceChooser(true)
                                )
                            );
                });

            Assert.IsTrue(eq);
        }
    
    
    
    }
}
