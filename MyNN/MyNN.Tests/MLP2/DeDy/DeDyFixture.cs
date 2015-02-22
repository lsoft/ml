using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.NextLayerAggregator;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.DeDy
{
    [TestClass]
    public class DeDyFixture
    {
        [TestMethod]
        public void Test0()
        {
            var rnd = new DefaultRandomizer(56);

            for (var seed = 0; seed < 1000; seed+=rnd.Next(25))
            {
                int previousLayerNeuronCount = 315 + rnd.Next(1500);
                int aggregateLayerNeuronCount = 315 + rnd.Next(1500);

                Console.WriteLine(
                    "seed = {0}, previousLayerNeuronCount = {1}, aggregateLayerNeuronCount = {2}",
                    seed,
                    previousLayerNeuronCount,
                    aggregateLayerNeuronCount
                    );

                var csharp = CalculateCSharp(
                    seed,
                    previousLayerNeuronCount,
                    aggregateLayerNeuronCount
                    );

                var cl = CalculateOpenCL(
                    seed,
                    previousLayerNeuronCount,
                    aggregateLayerNeuronCount
                    );

                int maxDiffIndex;
                float maxDiff;
                if (!ArrayOperations.ValuesAreEqual(csharp, cl, 1e-6f, out maxDiff, out maxDiffIndex))
                {
                    Console.WriteLine(
                        "Diff {0} at index {1}",
                        maxDiffIndex,
                        maxDiff
                        );

                    Assert.Fail();
                }
            }
        }

        private float[] CalculateCSharp(
            int seed,
            int previousLayerNeuronCount,
            int aggregateLayerNeuronCount
            )
        {
            var randomizer = new DefaultRandomizer(seed);

            var dedz = new float[aggregateLayerNeuronCount];
            var weights = new float[previousLayerNeuronCount * aggregateLayerNeuronCount];

            dedz.Fill((index) => randomizer.Next(5));
            weights.Fill((index) => randomizer.Next(5));

            var csharp = new CSharpDeDyCalculator(
                previousLayerNeuronCount,
                aggregateLayerNeuronCount,
                dedz,
                weights
                );

            csharp.ClearAndWrite();
            csharp.Aggregate();

            var csharpResults = csharp.DeDy.CloneArray();

            return
                csharpResults;
        }


        private float[] CalculateOpenCL(
            int seed,
            int previousLayerNeuronCount,
            int aggregateLayerNeuronCount
            )
        {
            var randomizer = new DefaultRandomizer(seed);

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), true))
            {
                var dedz = clProvider.CreateFloatMem(
                    aggregateLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                var weights = clProvider.CreateFloatMem(
                    previousLayerNeuronCount * aggregateLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                dedz.Array.Fill((index) => randomizer.Next(5));
                weights.Array.Fill((index) => randomizer.Next(5));

                dedz.Write(BlockModeEnum.Blocking);
                weights.Write(BlockModeEnum.Blocking);

                var cl = new OpenCLDeDyCalculator(
                    clProvider,
                    previousLayerNeuronCount,
                    aggregateLayerNeuronCount,
                    dedz,
                    weights
                    );

                cl.ClearAndWrite();
                cl.Aggregate();

                clProvider.QueueFinish();

                cl.DeDy.Read(BlockModeEnum.Blocking);

                var clResults = cl.DeDy.Array.GetSubArray(0, previousLayerNeuronCount);

                return
                    clResults;
            }
        }
    }
}
