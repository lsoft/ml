using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.Randomizer;
using MyNN.OutputConsole;
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

            TestWarpReduction(deviceChooser);
        }

        [TestMethod]
        public void IntelGPUWarpReductionTest()
        {
            var deviceChooser = new IntelGPUDeviceChooser();

            TestWarpReduction(deviceChooser);
        }

        [TestMethod]
        public void IntelCPUWarpReductionTest()
        {
            var deviceChooser = new IntelCPUDeviceChooser();

            TestWarpReduction(deviceChooser);
        }

        private static void TestWarpReduction(
            IDeviceChooser deviceChooser)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }

            var seed = DateTime.Now.Millisecond;
            var randomizer = new DefaultRandomizer(ref seed);

            for (var size = 2; size < 64; size++)
            {
                ConsoleAmbientContext.Console.WriteLine("size = {0}", size);

                using (var clProvider = new CLProvider(deviceChooser, true))
                {
                    var m = clProvider.CreateFloatMem(
                        size,
                        MemFlags.CopyHostPtr | MemFlags.ReadWrite);

                    for (var cc = 0; cc < m.Array.Length; cc++)
                    {
                        m.Array[cc] = randomizer.Next(256)/256f;
                    }

                    m.Write(BlockModeEnum.Blocking);

                    var cpuSum = m.Array.Sum();

                    var k = @"
__kernel void TestReductionKernel(
    __global float * gdata,
    __local float * ldata)
{
    ldata[get_local_id(0)] = gdata[get_local_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);
    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    WarpReductionToFirstElement(ldata);

    barrier(CLK_LOCAL_MEM_FENCE);
    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

//    ldata[0] = 0;
//    for(int a = 0; a < get_global_size(0); a++)
//        ldata[0] += gdata[a];

    if(get_local_id(0) == 0)
    {
        gdata[0] = ldata[0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
}
";

                    var kernel = clProvider.CreateKernel(k, "TestReductionKernel");

                    kernel
                        .SetKernelArgMem(0, m)
                        .SetKernelArgLocalMem(1, size)
                        .EnqueueNDRangeKernel(
                            new[] {size},
                            new int[] {size});

                    m.Read(BlockModeEnum.Blocking);

                    var gpuSum = m.Array[0];

                    Assert.AreEqual(cpuSum, gpuSum);
                }
            }
        }
    }
}
