using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.MLP.DropConnect.ForwardPropagation.WeightMaskContainer2;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.MaskContainers2
{
    [TestClass]
    public class BigArrayFixture
    {
        [TestMethod]
        public void Test05()
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

            var randomizer = new DefaultRandomizer(123);
            Func<CLProvider, IOpenCLWeightMaskContainer2> containerProvider =
                (clProvider) =>
                {
                    return
                        new BigArrayWeightMaskContainer2(
                            clProvider,
                            mlpConfiguration.Layers[1],
                            mlpConfiguration.Layers[2],
                            randomizer,
                            p);
                };

            var result = MaskContainerTester.TestContainer(
                containerProvider
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
        public void Test10()
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

            var randomizer = new DefaultRandomizer(123);
            Func<CLProvider, IOpenCLWeightMaskContainer2> containerProvider =
                (clProvider) =>
                {
                    return
                        new BigArrayWeightMaskContainer2(
                            clProvider,
                            mlpConfiguration.Layers[1],
                            mlpConfiguration.Layers[2],
                            randomizer,
                            p);
                };

            var result = MaskContainerTester.TestContainer(
                containerProvider
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
        public void Test01()
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

            var randomizer = new DefaultRandomizer(123);
            Func<CLProvider, IOpenCLWeightMaskContainer2> containerProvider =
                (clProvider) =>
                {
                    return
                        new BigArrayWeightMaskContainer2(
                            clProvider,
                            mlpConfiguration.Layers[1],
                            mlpConfiguration.Layers[2],
                            randomizer,
                            p);
                };

            var result = MaskContainerTester.TestContainer(
                containerProvider
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
        public void Test09()
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

            var randomizer = new DefaultRandomizer(123);
            Func<CLProvider, IOpenCLWeightMaskContainer2> containerProvider =
                (clProvider) =>
                {
                    return
                        new BigArrayWeightMaskContainer2(
                            clProvider,
                            mlpConfiguration.Layers[1],
                            mlpConfiguration.Layers[2],
                            randomizer,
                            p);
                };

            var result = MaskContainerTester.TestContainer(
                containerProvider
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
