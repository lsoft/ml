using System;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.DeDyAggregator;
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
        public void TestCPU()
        {
            var rnd = new DefaultRandomizer(56);

            for (var seed = 0; seed < 1000; seed += rnd.Next(25))
            {
                int previousLayerNeuronCount = 7 + rnd.Next(1500);
                int aggregateLayerNeuronCount = 3 + rnd.Next(1500);

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

                var cl = CalculateCPU(
                    seed,
                    previousLayerNeuronCount,
                    aggregateLayerNeuronCount
                    );

                int maxDiffIndex;
                float maxDiff;
                if (!ArrayOperations.ValuesAreEqual(csharp, cl, 1e-9f, out maxDiff, out maxDiffIndex))
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

        [TestMethod]
        public void TestGPU()
        {
            var rnd = new DefaultRandomizer(56);

            for (var seed = 0; seed < 1000; seed+=rnd.Next(25))
            {
                int previousLayerNeuronCount = 7 + rnd.Next(1500);
                int aggregateLayerNeuronCount = 3 + rnd.Next(1500);

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

                var cl = CalculateGPU(
                    seed,
                    previousLayerNeuronCount,
                    aggregateLayerNeuronCount
                    );

                int maxDiffIndex;
                float maxDiff;
                if (!ArrayOperations.ValuesAreEqual(csharp, cl, 1e-9f, out maxDiff, out maxDiffIndex))
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

            var weights = new float[previousLayerNeuronCount * aggregateLayerNeuronCount];

            weights.Fill((index) => randomizer.Next(5));

            var csharp = new CSharpDeDyAggregator(
                previousLayerNeuronCount,
                aggregateLayerNeuronCount,
                weights
                );

            csharp.ClearAndWrite();

            csharp.DeDz.Fill((index) => randomizer.Next(5));

            csharp.Aggregate();

            var csharpResults = csharp.DeDy.CloneArray();

            return
                csharpResults;
        }


        private float[] CalculateCPU(
            int seed,
            int previousLayerNeuronCount,
            int aggregateLayerNeuronCount
            )
        {
            var randomizer = new DefaultRandomizer(seed);

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(false), true))
            {
                var weights = clProvider.CreateFloatMem(
                    previousLayerNeuronCount * aggregateLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                var cpuAggregator = new CPUDeDyAggregator(
                    clProvider,
                    previousLayerNeuronCount,
                    aggregateLayerNeuronCount,
                    weights
                    );

                cpuAggregator.ClearAndWrite();

                weights.Array.Fill((index) => randomizer.Next(5));
                cpuAggregator.DeDz.Array.Fill((index) => randomizer.Next(5));

                cpuAggregator.DeDz.Write(BlockModeEnum.Blocking);
                weights.Write(BlockModeEnum.Blocking);

                cpuAggregator.Aggregate();

                clProvider.QueueFinish();

                cpuAggregator.DeDy.Read(BlockModeEnum.Blocking);

                var clResults = cpuAggregator.DeDy.Array.GetSubArray(0, previousLayerNeuronCount);

                return
                    clResults;
            }
        }

        private float[] CalculateGPU(
            int seed,
            int previousLayerNeuronCount,
            int aggregateLayerNeuronCount
            )
        {
            var randomizer = new DefaultRandomizer(seed);

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), true))
            {
                var weights = clProvider.CreateFloatMem(
                    previousLayerNeuronCount * aggregateLayerNeuronCount,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                var gpuAggregator = new GPUDeDyAggregator(
                    clProvider,
                    previousLayerNeuronCount,
                    aggregateLayerNeuronCount,
                    weights
                    );

                gpuAggregator.ClearAndWrite();

                weights.Array.Fill((index) => randomizer.Next(5));
                gpuAggregator.DeDz.Array.Fill((index) => randomizer.Next(5));

                gpuAggregator.DeDz.Write(BlockModeEnum.Blocking);
                weights.Write(BlockModeEnum.Blocking);
                
                gpuAggregator.Aggregate();

                clProvider.QueueFinish();

                gpuAggregator.DeDy.Read(BlockModeEnum.Blocking);

                var clResults = gpuAggregator.DeDy.Array.GetSubArray(0, previousLayerNeuronCount);

                return
                    clResults;
            }
        }
    }
}
