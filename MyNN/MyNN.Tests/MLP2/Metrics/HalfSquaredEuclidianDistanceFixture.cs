using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Backpropagation.Metrics;

namespace MyNN.Tests.MLP2.Metrics
{
    [TestClass]
    public class HalfSquaredEuclidianDistanceFixture
    {
        [TestMethod]
        public void Test()
        {
            var metric = new HalfSquaredEuclidianDistance();

            var mt = new MetricTester();
            mt.Test(
                metric
                );
        }
    }
}
