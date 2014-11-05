using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP.Backpropagation.Metrics;

namespace MyNN.Tests.MLP2.Metrics
{
    [TestClass]
    public class RMSEFixture
    {
        [TestMethod]
        public void Test()
        {
            var metric = new RMSE();

            var mt = new MetricTester();
            mt.Test(
                metric
                );
        }
    }
}