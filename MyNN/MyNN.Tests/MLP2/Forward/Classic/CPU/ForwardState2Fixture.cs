using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation.Classic;
using MyNN.MLP2.ForwardPropagation.Classic.CSharp;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.Mem.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.Forward.Classic.CPU
{
    /// <summary>
    /// Summary description for ForwardState2Fixture
    /// </summary>
    [TestClass]
    public class ForwardState2Fixture
    {
        private const float ForwardEpsilon = 1e-6f;

        public ForwardState2Fixture()
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
        public void Forward_1_1_1_NoVec_Test0()
        {
            var test = new ForwardStateTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });

            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_1_MLP(
                    dataset,
                    new List<float> {1f, 1f},
                    new List<float> {1f, 1f},
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            VectorizationSizeEnum.NoVectorization
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators);

                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                var correctResult = new Pair<float, float>(1.75f, 2.75f);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = [{0}; {1}], result = [{2};{3}]",
                        correctResult.First,
                        correctResult.Second,
                        result.First,
                        result.Second
                        ));

                Assert.IsTrue(Math.Abs(result.First - correctResult.First) < ForwardEpsilon);
                Assert.IsTrue(Math.Abs(result.Second - correctResult.Second) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_1_1_1_Vec4_Test0()
        {
            var test = new ForwardStateTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });

            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_1_MLP(
                    dataset,
                    new List<float> { 1f, 1f },
                    new List<float> { 1f, 1f },
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            VectorizationSizeEnum.VectorizationMode4
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators);

                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                var correctResult = new Pair<float, float>(1.75f, 2.75f);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = [{0}; {1}], result = [{2};{3}]",
                        correctResult.First,
                        correctResult.Second,
                        result.First,
                        result.Second
                        ));

                Assert.IsTrue(Math.Abs(result.First - correctResult.First) < ForwardEpsilon);
                Assert.IsTrue(Math.Abs(result.Second - correctResult.Second) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_1_1_1_Vec16_Test0()
        {
            var test = new ForwardStateTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });

            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_1_MLP(
                    dataset,
                    new List<float> { 1f, 1f },
                    new List<float> { 1f, 1f },
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new CPUPropagatorComponentConstructor(
                            clProvider,
                            VectorizationSizeEnum.VectorizationMode16
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators);

                        return
                            new ForwardPropagation2(
                                containers,
                                propagators,
                                mlp
                                );
                    });

                var correctResult = new Pair<float, float>(1.75f, 2.75f);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = [{0}; {1}], result = [{2};{3}]",
                        correctResult.First,
                        correctResult.Second,
                        result.First,
                        result.Second
                        ));

                Assert.IsTrue(Math.Abs(result.First - correctResult.First) < ForwardEpsilon);
                Assert.IsTrue(Math.Abs(result.Second - correctResult.Second) < ForwardEpsilon);
            }
        }




    }

}
