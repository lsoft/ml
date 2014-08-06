using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.Tests.MLP2.MaskContainers.BigArray
{
    [TestClass]
    public class NVidiaGPUFixture
    {
        [TestMethod]
        public void Test05WithCPU()
        {
            var mlpConfiguration = MLPConfigurationConstructor.CreateConfiguration(
                new int[]
                {
                    5,
                    5,
                    5
                });

            var p = 0.5f;
            var correctResult = new diapfloat(p, p/100f);

            var result = BitMaskContainerTester.TestContainer(
                new NvidiaOrAmdGPUDeviceChooser(), 
                mlpConfiguration,
                p
                );

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Result = {0}, correct result = {1}",
                    result,
                    correctResult
                    ));

            Assert.IsTrue(correctResult.IsValueInclusive(result));
        }

        [TestMethod]
        public void Test10WithCPU()
        {
            var mlpConfiguration = MLPConfigurationConstructor.CreateConfiguration(
                new int[]
                {
                    5,
                    5,
                    5
                });

            var p = 1f;
            var correctResult = new diapfloat(p, p, false);

            var result = BitMaskContainerTester.TestContainer(
                new NvidiaOrAmdGPUDeviceChooser(),
                mlpConfiguration,
                p
                );

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Result = {0}, correct result = {1}",
                    result,
                    correctResult
                    ));

            Assert.IsTrue(correctResult.IsValueInclusive(result));
        }

        [TestMethod]
        public void Test01WithCPU()
        {
            var mlpConfiguration = MLPConfigurationConstructor.CreateConfiguration(
                new int[]
                {
                    5,
                    5,
                    5
                });

            var p = 0.1f;
            var correctResult = new diapfloat(p, p / 100f);

            var result = BitMaskContainerTester.TestContainer(
                new NvidiaOrAmdGPUDeviceChooser(),
                mlpConfiguration,
                p
                );

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Result = {0}, correct result = {1}",
                    result,
                    correctResult
                    ));

            Assert.IsTrue(correctResult.IsValueInclusive(result));
        }

        [TestMethod]
        public void Test09WithCPU()
        {
            var mlpConfiguration = MLPConfigurationConstructor.CreateConfiguration(
                new int[]
                {
                    5,
                    5,
                    5
                });

            var p = 0.9f;
            var correctResult = new diapfloat(p, p / 100f);

            var result = BitMaskContainerTester.TestContainer(
                new NvidiaOrAmdGPUDeviceChooser(),
                mlpConfiguration,
                p
                );

            ConsoleAmbientContext.Console.WriteLine(
                string.Format(
                    "Result = {0}, correct result = {1}",
                    result,
                    correctResult
                    ));

            Assert.IsTrue(correctResult.IsValueInclusive(result));
        }
    }
}
