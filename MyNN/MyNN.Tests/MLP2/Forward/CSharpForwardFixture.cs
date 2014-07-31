using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation.Classic.CSharp;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using OpenCL.Net.Wrapper;

namespace MyNN.Tests.MLP2.Forward
{
    [TestClass]
    public class CSharpForwardFixture
    {
        private const float ForwardEpsilon = 1e-6f;

        [TestMethod]
        public void CSharpForward_1_1_Test0()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {0.75f},
                        new[] {1f})
                });

            var result = test.ExecuteTestWith_1_1_MLP(
                dataset,
                1f,
                1f,
                () => new LinearFunction(1f),
                (mlp) =>
                {
                    return
                        new CSharpForwardPropagation(mlp);
                });

            Assert.IsTrue(Math.Abs(result - 1.75f) < ForwardEpsilon);
        }

        [TestMethod]
        public void CSharpForward_1_1_Test1()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {2f},
                        new[] {1f})
                });
            
            var result = test.ExecuteTestWith_1_1_MLP(
                dataset,
                0.5f,
                -1f,
                () => new SigmoidFunction(1f),
                (mlp) =>
                {
                    return
                        new CSharpForwardPropagation(mlp);
                });

            Assert.IsTrue(Math.Abs(result - 0.5f) < ForwardEpsilon);
        }

        [TestMethod]
        public void CSharpForward_5_24_24_1_Test0()
        {
            var test = new ForwardTester();

            var dataset = new DataSet(
                new List<DataItem>
                {
                    new DataItem(
                        new[] {-0.2f, -0.1f, 0.1f, 0.3f, 0.8f},
                        new[] {1f})
                });
            
            var result = test.ExecuteTestWith_5_24_24_1_MLP(
                dataset,
                () => new LinearFunction(1f), 
                (mlp) =>
                {
                    return
                        new CSharpForwardPropagation(mlp);
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
}
