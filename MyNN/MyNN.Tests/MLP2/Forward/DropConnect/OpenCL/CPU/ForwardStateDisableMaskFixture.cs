namespace MyNN.Tests.MLP2.Forward.DropConnect.OpenCL.CPU
{
    //!!! починить!
    //[TestClass]
    //public class ForwardStateDisableMaskFixture
    //{
    //    private const float ForwardEpsilon = 1e-6f;

    //    public ForwardStateDisableMaskFixture()
    //    {
    //        //
    //        // TODO: Add constructor logic here
    //        //
    //    }

    //    private TestContext testContextInstance;

    //    /// <summary>
    //    ///Gets or sets the test context which provides
    //    ///information about and functionality for the current test run.
    //    ///</summary>
    //    public TestContext TestContext
    //    {
    //        get
    //        {
    //            return testContextInstance;
    //        }
    //        set
    //        {
    //            testContextInstance = value;
    //        }
    //    }

    //    #region Additional test attributes
    //    //
    //    // You can use the following additional attributes as you write your tests:
    //    //
    //    // Use ClassInitialize to run code before running the first test in the class
    //    // [ClassInitialize()]
    //    // public static void MyClassInitialize(TestContext testContext) { }
    //    //
    //    // Use ClassCleanup to run code after all tests in a class have run
    //    // [ClassCleanup()]
    //    // public static void MyClassCleanup() { }
    //    //
    //    // Use TestInitialize to run code before running each test 
    //    // [TestInitialize()]
    //    // public void MyTestInitialize() { }
    //    //
    //    // Use TestCleanup to run code after each test has run
    //    // [TestCleanup()]
    //    // public void MyTestCleanup() { }
    //    //
    //    #endregion

    //    [TestMethod]
    //    public void Forward_1_2_2_NoVec_Test0()
    //    {
    //        var test = new ForwardStateTester();

    //        var dataset = new TestDataSet(
    //            new List<IDataItem>
    //            {
    //                new DataItem(
    //                    new[] {0.75f},
    //                    new[] {1f})
    //            });

    //        using (var clProvider = new CLProvider())
    //        {
    //            var result = test.ExecuteTestWith_1_2_2_MLP(
    //                dataset,
    //                new List<float> {2f, 1f, 1f, 1f},
    //                new List<float> {4f, 1f, 1f, 2f, 2f, 1f},
    //                () => new LinearFunction(1f),
    //                (mlp) =>
    //                {
    //                    var maskContainer = new MockMaskContainer(
    //                        clProvider,
    //                        mlp,
    //                        1,
    //                        new uint[]
    //                        {
    //                            1,
    //                            1,
    //                            1
    //                        }
    //                        );

    //                    var forward = new DropConnectOpenCLForwardPropagation(
    //                        VectorizationSizeEnum.NoVectorization,
    //                        mlp,
    //                        clProvider,
    //                        maskContainer
    //                        );

    //                    return
    //                        forward;
    //                });

    //            var layer0CorrectResult = new Pair<float, float>(2.5f, 1.75f);
    //            ConsoleAmbientContext.Console.WriteLine(
    //                string.Format(
    //                    "layer 0: correct = [{0}; {1}], result = [{2};{3}]",
    //                    layer0CorrectResult.First,
    //                    layer0CorrectResult.Second,
    //                    result.First.First,
    //                    result.First.Second
    //                    ));

    //            var layer1CorrectResult = new Pair<float, float>(12.75f, 9.5f);
    //            ConsoleAmbientContext.Console.WriteLine(
    //                string.Format(
    //                    "layer 1: correct = [{0}; {1}], result = [{2};{3}]",
    //                    layer1CorrectResult.First,
    //                    layer1CorrectResult.Second,
    //                    result.Second.First,
    //                    result.Second.Second
    //                    ));

    //            Assert.IsTrue(Math.Abs(result.First.First - layer0CorrectResult.First) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.First.Second - layer0CorrectResult.Second) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.Second.First - layer1CorrectResult.First) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.Second.Second - layer1CorrectResult.Second) < ForwardEpsilon);
    //        }
    //    }

    //    [TestMethod]
    //    public void Forward_1_2_2_Vec4_Test0()
    //    {
    //        var test = new ForwardStateTester();

    //        var dataset = new TestDataSet(
    //            new List<IDataItem>
    //            {
    //                new DataItem(
    //                    new[] {0.75f},
    //                    new[] {1f})
    //            });

    //        using (var clProvider = new CLProvider())
    //        {
    //            var result = test.ExecuteTestWith_1_2_2_MLP(
    //                dataset,
    //                new List<float> { 2f, 1f, 1f, 1f },
    //                new List<float> { 4f, 1f, 1f, 2f, 2f, 1f },
    //                () => new LinearFunction(1f),
    //                (mlp) =>
    //                {
    //                    var maskContainer = new MockMaskContainer(
    //                        clProvider,
    //                        mlp,
    //                        1,
    //                        new uint[]
    //                        {
    //                            1,
    //                            1,
    //                            1
    //                        }
    //                        );

    //                    var forward = new DropConnectOpenCLForwardPropagation(
    //                        VectorizationSizeEnum.VectorizationMode4,
    //                        mlp,
    //                        clProvider,
    //                        maskContainer
    //                        );

    //                    return
    //                        forward;
    //                });

    //            var layer0CorrectResult = new Pair<float, float>(2.5f, 1.75f);
    //            ConsoleAmbientContext.Console.WriteLine(
    //                string.Format(
    //                    "layer 0: correct = [{0}; {1}], result = [{2};{3}]",
    //                    layer0CorrectResult.First,
    //                    layer0CorrectResult.Second,
    //                    result.First.First,
    //                    result.First.Second
    //                    ));

    //            var layer1CorrectResult = new Pair<float, float>(12.75f, 9.5f);
    //            ConsoleAmbientContext.Console.WriteLine(
    //                string.Format(
    //                    "layer 1: correct = [{0}; {1}], result = [{2};{3}]",
    //                    layer1CorrectResult.First,
    //                    layer1CorrectResult.Second,
    //                    result.Second.First,
    //                    result.Second.Second
    //                    ));

    //            Assert.IsTrue(Math.Abs(result.First.First - layer0CorrectResult.First) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.First.Second - layer0CorrectResult.Second) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.Second.First - layer1CorrectResult.First) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.Second.Second - layer1CorrectResult.Second) < ForwardEpsilon);
    //        }
    //    }


    //    [TestMethod]
    //    public void Forward_1_2_2_Vec16_Test0()
    //    {
    //        var test = new ForwardStateTester();

    //        var dataset = new TestDataSet(
    //            new List<IDataItem>
    //            {
    //                new DataItem(
    //                    new[] {0.75f},
    //                    new[] {1f})
    //            });

    //        using (var clProvider = new CLProvider())
    //        {
    //            var result = test.ExecuteTestWith_1_2_2_MLP(
    //                dataset,
    //                new List<float> { 2f, 1f, 1f, 1f },
    //                new List<float> { 4f, 1f, 1f, 2f, 2f, 1f },
    //                () => new LinearFunction(1f),
    //                (mlp) =>
    //                {
    //                    var maskContainer = new MockMaskContainer(
    //                        clProvider,
    //                        mlp,
    //                        1,
    //                        new uint[]
    //                        {
    //                            1,
    //                            1,
    //                            1
    //                        }
    //                        );

    //                    var forward = new DropConnectOpenCLForwardPropagation(
    //                        VectorizationSizeEnum.VectorizationMode16,
    //                        mlp,
    //                        clProvider,
    //                        maskContainer
    //                        );

    //                    return
    //                        forward;
    //                });

    //            var layer0CorrectResult = new Pair<float, float>(2.5f, 1.75f);
    //            ConsoleAmbientContext.Console.WriteLine(
    //                string.Format(
    //                    "layer 0: correct = [{0}; {1}], result = [{2};{3}]",
    //                    layer0CorrectResult.First,
    //                    layer0CorrectResult.Second,
    //                    result.First.First,
    //                    result.First.Second
    //                    ));

    //            var layer1CorrectResult = new Pair<float, float>(12.75f, 9.5f);
    //            ConsoleAmbientContext.Console.WriteLine(
    //                string.Format(
    //                    "layer 1: correct = [{0}; {1}], result = [{2};{3}]",
    //                    layer1CorrectResult.First,
    //                    layer1CorrectResult.Second,
    //                    result.Second.First,
    //                    result.Second.Second
    //                    ));

    //            Assert.IsTrue(Math.Abs(result.First.First - layer0CorrectResult.First) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.First.Second - layer0CorrectResult.Second) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.Second.First - layer1CorrectResult.First) < ForwardEpsilon);
    //            Assert.IsTrue(Math.Abs(result.Second.Second - layer1CorrectResult.Second) < ForwardEpsilon);
    //        }
    //    }



    //}

}
