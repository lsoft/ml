using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU.Two;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.Forward
{
    /// <summary>
    /// Summary description for OpenCLForwardFixture
    /// </summary>
    [TestClass]
    public class OpenCLForward2Fixture
    {
        private const float ForwardEpsilon = 1e-6f;

        public OpenCLForward2Fixture()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        private TestContext testContextInstance;

        /// <summary>
        ///Gets or sets the test context which provides
        ///information about and functionality for the current test run.
        ///</summary>
        public TestContext TestContext
        {
            get
            {
                return testContextInstance;
            }
            set
            {
                testContextInstance = value;
            }
        }

        #region Additional test attributes
        //
        // You can use the following additional attributes as you write your tests:
        //
        // Use ClassInitialize to run code before running the first test in the class
        // [ClassInitialize()]
        // public static void MyClassInitialize(TestContext testContext) { }
        //
        // Use ClassCleanup to run code after all tests in a class have run
        // [ClassCleanup()]
        // public static void MyClassCleanup() { }
        //
        // Use TestInitialize to run code before running each test 
        // [TestInitialize()]
        // public void MyTestInitialize() { }
        //
        // Use TestCleanup to run code after each test has run
        // [TestCleanup()]
        // public void MyTestCleanup() { }
        //
        #endregion

        [TestMethod]
        public void OpenCLForward_1_1_Test0()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_MLP(
                    dataset,
                    1f,
                    1f,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            mlp);

                        IMemLayerContainer[] containers;
                        ICPULayerPropagator[] propagators;
                        pcc.CreateComponents(
                            VectorizationSizeEnum.NoVectorization,
                            out containers,
                            out propagators);

                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                Assert.IsTrue(Math.Abs(result - 1.75f) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void OpenCLForward_1_1_Test1()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {2f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_MLP(
                    dataset,
                    0.5f,
                    -1f,
                    () => new SigmoidFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            mlp);

                        IMemLayerContainer[] containers;
                        ICPULayerPropagator[] propagators;
                        pcc.CreateComponents(
                            VectorizationSizeEnum.NoVectorization,
                            out containers,
                            out propagators);

                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                Assert.IsTrue(Math.Abs(result - 0.5f) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void OpenCLForward_1_1_Test2()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {2f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_MLP(
                    dataset,
                    0.5f,
                    -1f,
                    () => new SigmoidFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            mlp);

                        IMemLayerContainer[] containers;
                        ICPULayerPropagator[] propagators;
                        pcc.CreateComponents(
                            VectorizationSizeEnum.VectorizationMode4,
                            out containers,
                            out propagators);


                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                Assert.IsTrue(Math.Abs(result - 0.5f) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void OpenCLForward_1_1_Test3()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {2f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_MLP(
                    dataset,
                    0.5f,
                    -1f,
                    () => new SigmoidFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            mlp);

                        IMemLayerContainer[] containers;
                        ICPULayerPropagator[] propagators;
                        pcc.CreateComponents(
                            VectorizationSizeEnum.VectorizationMode16,
                            out containers,
                            out propagators);

                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                Assert.IsTrue(Math.Abs(result - 0.5f) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void OpenCLForward_5_24_24_1_Test0()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_24_24_1_MLP(
                    dataset,
                    () => new LinearFunction(1f), 
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            mlp);

                        IMemLayerContainer[] containers;
                        ICPULayerPropagator[] propagators;
                        pcc.CreateComponents(
                            VectorizationSizeEnum.VectorizationMode16,
                            out containers,
                            out propagators);


                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                const float correctResult = -0.4017001f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void OpenCLForward_5_24_24_1_Test1()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_24_24_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            mlp);

                        IMemLayerContainer[] containers;
                        ICPULayerPropagator[] propagators;
                        pcc.CreateComponents(
                            VectorizationSizeEnum.VectorizationMode4,
                            out containers,
                            out propagators);


                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                const float correctResult = -0.4017001f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void OpenCLForward_5_24_24_1_Test2()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_24_24_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            mlp);

                        IMemLayerContainer[] containers;
                        ICPULayerPropagator[] propagators;
                        pcc.CreateComponents(
                            VectorizationSizeEnum.NoVectorization,
                            out containers,
                            out propagators);

                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                const float correctResult = -0.4017001f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }


    }
}
