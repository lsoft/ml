﻿using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU;
using MyNN.MLP.DeDyAggregator;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.Forward.Classic.CPU
{
    /// <summary>
    /// Summary description for ForwardOutput2Fixture
    /// </summary>
    [TestClass]
    public class ForwardOutput2Fixture
    {
        private const float ForwardEpsilon = 1e-6f;

        public ForwardOutput2Fixture()
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
        public void Forward_1_1_NoVec_Test0()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
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
                            VectorizationSizeEnum.NoVectorization
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                const float correctResult = 1.75f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_1_1_NoVec_Test1()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
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
                            VectorizationSizeEnum.NoVectorization
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                Assert.IsTrue(Math.Abs(result - 0.5f) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_1_1_Vec4_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
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
                            VectorizationSizeEnum.VectorizationMode4
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                Assert.IsTrue(Math.Abs(result - 0.5f) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_1_1_Vec16_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
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
                            VectorizationSizeEnum.VectorizationMode16
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                const float correctResult = 0.5f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_5_24_24_1_Vec16_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
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
                            VectorizationSizeEnum.VectorizationMode16
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
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
        public void Forward_5_24_24_1_Vec4_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
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
                            VectorizationSizeEnum.VectorizationMode4
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
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
        public void Forward_5_24_24_1_NoVec_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
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
                            VectorizationSizeEnum.NoVectorization
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
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
        public void Forward_5_300_1_NoVec_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_300_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            VectorizationSizeEnum.NoVectorization
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                var correctResult = new diapfloat(5.8f, 0.000015f);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(correctResult.IsValueInclusive(result));
            }
        }

        [TestMethod]
        public void Forward_5_300_1_Vec4_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_300_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            VectorizationSizeEnum.VectorizationMode4
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                var correctResult = new diapfloat(5.8f, 0.000005f);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(correctResult.IsValueInclusive(result));
            }
        }

        [TestMethod]
        public void Forward_5_300_1_Vec16_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_300_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            VectorizationSizeEnum.VectorizationMode16
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        IDeDyAggregator[] dedyAggregators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators,
                            out dedyAggregators
                            );

                        return
                            new ForwardPropagation(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                const float correctResult = 5.8f;

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
