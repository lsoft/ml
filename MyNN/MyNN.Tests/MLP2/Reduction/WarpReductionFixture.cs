using System;
using System.Diagnostics;
using System.Linq;
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

            TestReduction(deviceChooser, 5999, 79);
        }

        [TestMethod]
        public void IntelGPUWarpReductionTest()
        {
            var deviceChooser = new IntelGPUDeviceChooser();

            TestReduction(deviceChooser, 5999, 79);
        }

        [TestMethod]
        public void IntelCPUWarpReductionTest()
        {
            var deviceChooser = new IntelCPUDeviceChooser();
            
            TestReduction(deviceChooser, 1511, 61);
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
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096
            };

            using (var clProvider = new CLProvider(deviceChooser, false))
            {
                var k = @"
__kernel void TestReductionKernel(
    __global float * gdata,
    __local float * ldata,
    int size
)
{
    if(get_global_size(0) >= size)
    {
        return;
    }

    ldata[get_local_id(0)] = gdata[get_group_id(0) * get_local_size(0) + get_local_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    gdata[get_group_id(0) * get_local_size(0) + get_local_id(0)] = 0;

    WarpReductionToFirstElement(ldata);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0)
    {
        gdata[get_group_id(0)] = ldata[0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
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

                        ConsoleAmbientContext.Console.WriteLine(
                            "csize = {0}, gsize = {1}, lsize = {2}...    ",
                            currentSize,
                            globalSize,
                            localSize);

                        using (MemFloat m = clProvider.CreateFloatMem(
                            currentSize,
                            MemFlags.CopyHostPtr | MemFlags.ReadWrite))
                        {

                            for (var cc = 0; cc < m.Array.Length; cc++)
                            {
                                m.Array[cc] = randomizer.Next(256)/256f;
                            }

                            m.Write(BlockModeEnum.Blocking);

                            var cpuSum = m.Array.Sum();

                            kernel
                                .SetKernelArgMem(0, m)
                                .SetKernelArgLocalMem(1, currentSize)
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

                            var gpuSum = m.Array.Sum();

                            Assert.AreEqual(cpuSum, gpuSum);
                        }
                    }
                }
            }
        }
    }
}
