using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP.Structure.Neuron.Function;
using MyNN.Tests.MLP2.ActivationFunction.CSharp;

namespace MyNN.Tests.MLP2.ActivationFunction.OpenCL
{
    /// <summary>
    /// Summary description for HyperbolicTangensOpenCLFixture
    /// </summary>
    [TestClass]
    public class HyperbolicTangensOpenCLFixture
    {
        public HyperbolicTangensOpenCLFixture()
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
        public void HyperbolicTangensTestWithOneOneCoef_OpenCL()
        {
            var sf = new HyperbolicTangensFunction(1f, 1f);

            var tests = new ActivationFunctionDerivativeTests();
            tests.ExecuteTests(sf);
        }

        [TestMethod]
        public void HyperbolicTangensTestWithDefaultCoef_OpenCL()
        {
            var sf = new HyperbolicTangensFunction();

            var tests = new ActivationFunctionDerivativeOpenCLTests();
            tests.ExecuteTests(sf);
        }
    }
}
