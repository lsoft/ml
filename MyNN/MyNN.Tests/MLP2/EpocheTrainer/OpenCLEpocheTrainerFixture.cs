using System.Diagnostics;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Data;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.Tests.MLP2.EpocheTrainer
{
    /// <summary>
    /// Summary description for OpenCLEpocheTrainerFixture
    /// </summary>
    [TestClass]
    public class OpenCLEpocheTrainerFixture
    {
        public OpenCLEpocheTrainerFixture()
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
            var trainer = new EpocheTrainer_1_1_Test();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.5f},
                        new[] {2f})
                });

            trainer.ExecuteTest(
                dataset,
                1f,
                1f,
                () => new LinearFunction(1f));
        }
    }
}
