using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.Reduction
{
    [TestClass]
    public class WarpReductionPerformanceFixture
    {
        [TestMethod]
        public void NvidiaAMDWarpReductionTest()
        {
            var deviceChooser = new NvidiaOrAmdGPUDeviceChooser();

            TestReduction(deviceChooser, 150000, 20000000);
        }

        [TestMethod]
        public void IntelGPUWarpReductionTest()
        {
            var deviceChooser = new IntelGPUDeviceChooser();

            TestReduction(deviceChooser, 150000, 20000000);
        }

        [TestMethod]
        public void IntelCPUWarpReductionTest()
        {
            var deviceChooser = new IntelCPUDeviceChooser();

            TestReduction(deviceChooser, 150000, 20000000);
        }

        private static void TestReduction(
            IDeviceChooser deviceChooser,
            int testCount,
            int upperBound
            )
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }

            var seed = DateTime.Now.Millisecond;
            var randomizer = new DefaultRandomizer(seed);

            using (var clProvider = new CLProvider(deviceChooser, false))
            {
                var k = @"
__kernel void TestReductionKernel(
    __global float * gdata,
    __global float * tdata,
    __local float * ldata,
    int currentSize
)
{
    float v = 0;
    if(get_global_id(0) < currentSize)
    {
        for(int i = get_global_id(0); i < currentSize; i += get_global_size(0))
        {
            v += gdata[i];
        }
    }

    ldata[get_local_id(0)] = v;

    barrier(CLK_LOCAL_MEM_FENCE);

    WarpReductionToFirstElement(ldata);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0)
    {
        tdata[get_group_id(0)] = ldata[0];
    }
}
";

                var kernel = clProvider.CreateKernel(k, "TestReductionKernel");

                var maxGroupCount = (int)clProvider.Parameters.NumComputeUnits*32;

                var testParameters = new List<TestParameters>();

                //добавляем с минимальными значениями предопределенные поля
                for (var groupCount = 1; groupCount < 8; groupCount++)
                {
                    for (var localSize = 1; localSize < 8; localSize++)
                    {
                        for (var currentSize = 1; currentSize < 64; currentSize++)
                        {
                            testParameters.Add(
                                new TestParameters(
                                    groupCount,
                                    localSize,
                                    currentSize));
                        }
                    }
                }

                //добавляем со стандартными параметрами
                for (var testIndex = 0; testIndex < testCount; testIndex++)
                {
                    var groupCount = randomizer.Next(maxGroupCount) + 1;

                    var localMaxByDeviceMemory = (int) clProvider.Parameters.LocalMemorySize/4/sizeof (float)/groupCount;
                    var localMaxByWorkGroupSize = (int) clProvider.Parameters.MaxWorkGroupSize;
                    var localSize = randomizer.Next(Math.Min(localMaxByDeviceMemory - 1, localMaxByWorkGroupSize - 1)) + 1;

                    var currentSize = Math.Min(localSize, randomizer.Next(upperBound) + 1);
                    //var globalSize = groupCount*localSize;

                    //var groupCount = 1;
                    //var localSize = 2699;
                    //var currentSize = 8;
                    //var globalSize = groupCount * localSize;

                    testParameters.Add(
                        new TestParameters(
                            groupCount,
                            localSize,
                            currentSize));
                }

                foreach(var tp in testParameters)
                {
                    ConsoleAmbientContext.Console.Write(
                        "csize = {0}, gsize = {1}, lsize = {2}, gcount = {3}, overhead = {4}...    ",
                        tp.CurrentSize,
                        tp.GlobalSize,
                        tp.LocalSize,
                        tp.GroupCount,
                        tp.GlobalSize - tp.CurrentSize
                        );

                    using (var m = clProvider.CreateFloatMem(
                        tp.CurrentSize,
                        MemFlags.CopyHostPtr | MemFlags.ReadOnly))
                    using (var t = clProvider.CreateFloatMem(
                        tp.GroupCount,
                        MemFlags.CopyHostPtr | MemFlags.WriteOnly))
                    {

                        for (var cc = 0; cc < m.Array.Length; cc++)
                        {
                            m.Array[cc] = randomizer.Next(8) + 1;
                        }

                        m.Write(BlockModeEnum.Blocking);

                        Array.Clear(t.Array, 0, t.Array.Length);
                        t.Write(BlockModeEnum.Blocking);

                        var cpuSum = KahanAlgorithm.Reduce(m.Array);

                        kernel
                            .SetKernelArgMem(0, m)
                            .SetKernelArgMem(1, t)
                            .SetKernelArgLocalMem(2, tp.LocalSize * 4)
                            .SetKernelArg(3, 4, tp.CurrentSize)
                            .EnqueueNDRangeKernel(
                                new[]
                                    {
                                        tp.GlobalSize
                                    },
                                new[]
                                    {
                                        tp.LocalSize
                                    }
                            );

                        clProvider.QueueFinish();

                        t.Read(BlockModeEnum.Blocking);

                        var gpuSum = t.Array.Sum();

                        ConsoleAmbientContext.Console.WriteLine((cpuSum - gpuSum).ToString());

                        Assert.AreEqual(cpuSum, gpuSum);
                    }
                }
            }
        }

        private class TestParameters
        {
            public int GroupCount
            {
                get;
                private set;
            }

            public int LocalSize
            {
                get;
                private set;
            }

            public int CurrentSize
            {
                get;
                private set;
            }

            public int GlobalSize
            {
                get
                {
                    return
                        this.GroupCount*this.LocalSize;
                }
            }

            public TestParameters(int groupCount, int localSize, int currentSize)
            {
                GroupCount = groupCount;
                LocalSize = localSize;
                CurrentSize = currentSize;
            }
        }
    }
}
