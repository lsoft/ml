using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Classic.BackpropagationFactory.Classic.OpenCL.CPU;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.Tests.MLP2.EpocheTrainer
{
    /// <summary>
    /// Summary description for CPUEpocheTrainerFixture
    /// </summary>
    [TestClass]
    public class CPUEpocheTrainerFixture
    {
        public const double Epsilon = 1e-12;

        public CPUEpocheTrainerFixture()
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
        [ClassInitialize()]
        public static void MyClassInitialize(TestContext testContext)
        {
            ConsoleAmbientContext.ResetConsole();
            ConsoleAmbientContext.Console = new TestOutputConsole();
        }
        
        // Use ClassCleanup to run code after all tests in a class have run
        [ClassCleanup()]
        public static void MyClassCleanup()
        {
            ConsoleAmbientContext.ResetConsole();
        }
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
        public void TestMethod1()
        {
            const int InputLength = 10;
            const int OutputLength = 2;

            var dil = new List<IDataItem>();

            for (var cc = 0; cc < 10; cc++)
            {
                var input = new float[InputLength];
                input[cc % InputLength] = 1f;

                var output = new float[OutputLength];
                output[cc % OutputLength] = 1f;

                var di = new DataItem(input, output);
                dil.Add(di);
            }

            var dataset = new TestDataSet(dil);

            var csTrainer = new CSharpTestHelper();
            var csAcc = csTrainer.ExecuteTest(
                dataset,
                () => new LinearFunction(1f));

            var gpuTrainer = new CPUTestHelper();
            var gpuAcc = gpuTrainer.ExecuteTest(
                dataset,
                () => new LinearFunction(1f));

            var correctResult = (double)csAcc.PerItemError;
            var result = (double)gpuAcc.PerItemError;

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "correct = {0}, result = {1}",
                    correctResult,
                    result));

            Assert.IsTrue(Math.Abs(result - correctResult) < Epsilon);


        }

    }
}
