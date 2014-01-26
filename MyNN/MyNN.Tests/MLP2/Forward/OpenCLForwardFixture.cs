﻿using System;
using System.Diagnostics;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.Tests.MLP2.Forward
{
    /// <summary>
    /// Summary description for OpenCLForwardFixture
    /// </summary>
    [TestClass]
    public class OpenCLForwardFixture
    {
        private const float ForwardEpsilon = 1e-6f;

        public OpenCLForwardFixture()
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
            var test = new Forward_1_1_Test();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });


            var result = test.ExecuteTest(
                dataset,
                1f,
                1f,
                () => new LinearFunction(1f), 
                (clProvider, mlp) =>
                {
                    return 
                        new CPUForwardPropagation(
                            VectorizationSizeEnum.NoVectorization,
                            mlp,
                            clProvider);
                });

            Assert.IsTrue(Math.Abs(result - 1.75f) < ForwardEpsilon);
        }

        [TestMethod]
        public void OpenCLForward_1_1_Test1()
        {
            var test = new Forward_1_1_Test();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {2f},
                        new[] {1f})
                });


            var result = test.ExecuteTest(
                dataset,
                0.5f,
                -1f,
                () => new SigmoidFunction(1f), 
                (clProvider, mlp) =>
                {
                    return
                        new CPUForwardPropagation(
                            VectorizationSizeEnum.NoVectorization,
                            mlp,
                            clProvider);
                });

            Assert.IsTrue(Math.Abs(result - 0.5f) < ForwardEpsilon);
        }

    }
}
