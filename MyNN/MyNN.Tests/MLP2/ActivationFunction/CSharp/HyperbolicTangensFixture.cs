using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.Tests.MLP2.ActivationFunction.CSharp
{
    /// <summary>
    /// Summary description for HyperbolicTangensFixture
    /// </summary>
    [TestClass]
    public class HyperbolicTangensFixture
    {
        public HyperbolicTangensFixture()
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
        public void HyperbolicTangensTestWithOneOneCoef()
        {
            var sf = new HyperbolicTangensFunction(1f, 1f);

            var tests = new ActivationFunctionDerivativeTests();
            tests.ExecuteTests(sf);
        }

        [TestMethod]
        public void HyperbolicTangensTestWithDefaultCoef()
        {
            var sf = new HyperbolicTangensFunction();

            var tests = new ActivationFunctionDerivativeTests();
            tests.ExecuteTests(sf);
        }
    }
}
