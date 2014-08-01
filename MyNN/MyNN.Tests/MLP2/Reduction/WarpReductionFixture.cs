using System;
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
    public class WarpReductionFixture
    {
        [TestMethod]
        public void NvidiaAMDWarpReductionTest()
        {
            var deviceChooser = new NvidiaOrAmdGPUDeviceChooser();

            TestReduction(deviceChooser, 33000, 13);
        }

        [TestMethod]
        public void IntelGPUWarpReductionTest()
        {
            var deviceChooser = new IntelGPUDeviceChooser();

            TestReduction(deviceChooser, 33000, 13);
        }

        [TestMethod]
        public void IntelCPUWarpReductionTest()
        {
            var deviceChooser = new IntelCPUDeviceChooser();

            TestReduction(deviceChooser, 33000, 13);
        }

        private static void TestReduction(
            IDeviceChooser deviceChooser,
            int upperBound,
            int randomStep
            )
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }

            var seed = DateTime.Now.Millisecond;
            var randomizer = new DefaultRandomizer(seed);

            var localSizes = new[]
            {
                3,
                4,
                8,
                16,
                24,
                31,
                32,
                33,
                64,
                128,
                255,
                256,
                257,
                512,
                991,
                997,
                1024,
                1601,
                2048,
                3541,
                4095,
                4096,
            };

            //var localSizes = new[]
            //{
            //    4,
            //    8,
            //    16,
            //    32,
            //    64,
            //    128,
            //    256,
            //    512,
            //    1024,
            //    2048,
            //    4096,
            //};

            using (var clProvider = new CLProvider(deviceChooser, false))
            {
                var k = @"
__kernel void TestReductionKernel(
    __global float * gdata,
    __local float * ldata,
    int currentSize
)
{
    float v = 0;

    if(get_global_id(0) < currentSize)
    {
        v = gdata[get_global_id(0)];
    }
    
    ldata[get_local_id(0)] = v;

    barrier(CLK_LOCAL_MEM_FENCE);

    WarpReductionToFirstElement(ldata);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0)
    {
        gdata[get_group_id(0)] = ldata[0];
    }
}
";

                var kernel = clProvider.CreateKernel(k, "TestReductionKernel"); 
                
                for (var currentSize = localSizes.Min(); currentSize < upperBound; currentSize += (randomizer.Next(randomStep) + 1))
                {
                    foreach (var localSize in localSizes)
                    {
                        if (localSize > clProvider.Parameters.MaxWorkGroupSize)
                        {
                            continue;
                        }

                        var groupCount =
                            (currentSize/localSize) + ((currentSize%localSize) == 0 ? 0 : 1);

                        var globalSize = groupCount*localSize;

                        var localMemoryInBytes = (ulong)groupCount * (ulong)localSize * 4;
                        if (localMemoryInBytes > clProvider.Parameters.LocalMemorySize / 2)
                        {
                            //слишком жирно - кончится локальная память у устройства
                            continue;
                        }

                        ConsoleAmbientContext.Console.Write(
                            "csize = {0}, gsize = {1}, lsize = {2}, gcount = {3}, overhead = {4}...    ",
                            currentSize,
                            globalSize,
                            localSize,
                            groupCount,
                            globalSize - currentSize
                            );

                        using (var m = clProvider.CreateFloatMem(
                            currentSize,
                            MemFlags.CopyHostPtr | MemFlags.ReadWrite))
                        {

                            for (var cc = 0; cc < m.Array.Length; cc++)
                            {
                                m.Array[cc] = randomizer.Next(8) + 1;
                            }

                            m.Write(BlockModeEnum.Blocking);

                            var cpuSum = KahanAlgorithm.Reduce(m.Array);

                            kernel
                                .SetKernelArgMem(0, m)
                                .SetKernelArgLocalMem(1, localSize * 4)
                                .SetKernelArg(2, 4, currentSize)
                                .EnqueueNDRangeKernel(
                                    new[]
                                    {
                                        globalSize
                                    },
                                    new[]
                                    {
                                        localSize
                                    }
                                );

                            clProvider.QueueFinish();

                            m.Read(BlockModeEnum.Blocking);

                            var gpuSum = m.Array.Take(groupCount).Sum();

                            ConsoleAmbientContext.Console.WriteLine((cpuSum - gpuSum).ToString());

                            Assert.AreEqual(cpuSum, gpuSum);
                        }
                    }
                }
            }
        }


    }
}
