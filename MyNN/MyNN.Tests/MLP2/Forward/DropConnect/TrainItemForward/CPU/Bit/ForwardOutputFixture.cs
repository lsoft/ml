using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation.DropConnect.TrainItemForward.Bit.OpenCL.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Tests.MLP2.Forward.DropConnect.TrainItemForward.CPU.Bit.MaskContainer;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.Forward.DropConnect.TrainItemForward.CPU.Bit
{
    [TestClass]
    public class ForwardOutputFixture
    {
        private const float ForwardEpsilon = 1e-6f;

        [TestMethod]
        public void Forward_1_1_NoVec_Test0()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });

            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_MLP(
                    dataset,
                    1f,
                    1f,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.NoVectorization,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                const float correctResult = 1.75f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_1_1_Vec4_Test0()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });

            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_MLP(
                    dataset,
                    1f,
                    1f,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.VectorizationMode4,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                const float correctResult = 1.75f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_1_1_Vec16_Test0()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });

            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_1_1_MLP(
                    dataset,
                    1f,
                    1f,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.VectorizationMode16,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                const float correctResult = 1.75f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_5_24_24_1_NoVec_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_24_24_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.NoVectorization,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                const float correctResult = -0.4017001f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_5_24_24_1_Vec4_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_24_24_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.VectorizationMode4,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                const float correctResult = -0.4017001f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_5_24_24_1_Vec16_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_24_24_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.VectorizationMode16,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                const float correctResult = -0.4017001f;

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(Math.Abs(result - correctResult) < ForwardEpsilon);
            }
        }

        [TestMethod]
        public void Forward_5_300_1_NoVec_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_300_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.NoVectorization,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                var correctResult = new diapfloat(5.8f, 0.000015f);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(correctResult.IsValueInclusive(result));
            }
        }

        [TestMethod]
        public void Forward_5_300_1_Vec4_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_300_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.VectorizationMode4,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                var correctResult = new diapfloat(5.8f, 0.000015f);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(correctResult.IsValueInclusive(result));
            }
        }

        [TestMethod]
        public void Forward_5_300_1_Vec16_Test()
        {
            var test = new ForwardOutputTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });


            using (var clProvider = new CLProvider())
            {
                var result = test.ExecuteTestWith_5_300_1_MLP(
                    dataset,
                    () => new LinearFunction(1f),
                    (mlp) =>
                    {
                        var maskContainer = new MockWeightBitMaskContainer(
                            clProvider,
                            mlp,
                            1,
                            1);// no mask

                        var forward = new DropConnectBitOpenCLForwardPropagation(
                            VectorizationSizeEnum.VectorizationMode16,
                            mlp,
                            clProvider,
                            maskContainer
                            );

                        return
                            forward;
                    });

                var correctResult = new diapfloat(5.8f, 0.000015f);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "correct = {0}, result = {1}",
                        correctResult,
                        result));

                Assert.IsTrue(correctResult.IsValueInclusive(result));
            }
        }

    
    }
}
