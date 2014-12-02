using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Data;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.GPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.Tests.MLP2.Forward.Classic.GPU
{
    /// <summary>
    /// Summary description for ForwardState2ForIntelGPUFixture
    /// </summary>
    [TestClass]
    public class ForwardState2ForIntelGPUFixture
    {
        private const float ForwardEpsilon = 1e-6f;

        public ForwardState2ForIntelGPUFixture()
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
        public void Forward_1_1_1_Test0()
        {
            var test = new ForwardStateTester();

            var dataset = new TestDataSet(
                new List<IDataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });

            using (var clProvider = new CLProvider(new IntelGPUDeviceChooser(true), false))
            {
                var result = test.ExecuteTestWith_1_1_1_MLP(
                    dataset,
                    new List<float> {1f, 1f},
                    new List<float> {1f, 1f},
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var pcc = new GPUPropagatorComponentConstructor(
                            clProvider
                            );

                        ILayerContainer[] containers;
                        ILayerPropagator[] propagators;
                        pcc.CreateComponents(
                            mlp,
                            out containers,
                            out propagators);

                        return
                            new ForwardPropagation(
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
