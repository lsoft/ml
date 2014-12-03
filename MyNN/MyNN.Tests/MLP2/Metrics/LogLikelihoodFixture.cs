using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP.Backpropagation.Metrics;

namespace MyNN.Tests.MLP2.Metrics
{
    [TestClass]
    public class LogLikelihoodFixture
    {
        [TestMethod]
        public void Test()
        {
            var metric = new Loglikelihood();

            var mt = new MetricTester();
            mt.Test(
                metric,
                (random) => (float) (random.NextDouble() * 0.7f + 0.1f)
                );
        }
    }
}