using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.MLP.Backpropagation.Metrics;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.Metrics
{
    public class MetricTester
    {
        public void Test(
            IMetrics metric,
            Func<Random, float> randomProvider = null
            )
        {
            if (metric == null)
            {
                throw new ArgumentNullException("metric");
            }

            randomProvider = randomProvider ?? new Func<Random, float>((random) => (float) (random.NextDouble()*2 - 0.5f));

            const int length = 97;
            const string methodName = "CalculateMetric";
            const string kernelName = "CheckMetricKernel";
            const float epsilon = 1e-5f;

            using (var clProvider = new CLProvider())
            {
                var method = metric.GetOpenCLPartialDerivative(
                    methodName,
                    VectorizationSizeEnum.NoVectorization,
                    MemModifierEnum.Global,
                    length
                    );

                var kernel = CheckKernel;

                kernel = kernel.Replace(
                    "{KERNEL_NAME}",
                    kernelName
                    );

                kernel = kernel.Replace(
                    "{CALCULATE_METRIC_CALL}",
                    methodName
                    );

                kernel = kernel.Replace(
                    "{CALCULATE_METRIC_BODY}",
                    method
                    );

                var k = clProvider.CreateKernel(
                    kernel,
                    kernelName
                    );

                var v1 = clProvider.CreateFloatMem(
                    length,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                var v2 = clProvider.CreateFloatMem(
                    length,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                var result = clProvider.CreateFloatMem(
                    1,
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                var random = new Random(DateTime.Now.Millisecond);

                for (var cc = 0; cc < 10000; cc++)
                {
                    v1.Array.Fill(() => randomProvider(random));
                    v1.Write(BlockModeEnum.Blocking);

                    v2.Array.Fill(() => randomProvider(random));
                    v2.Write(BlockModeEnum.Blocking);

                    var v2Index = random.Next(length);

                    var csharpResult = metric.CalculatePartialDerivativeByV2Index(
                        v1.Array,
                        v2.Array,
                        v2Index
                        );

                    k
                        .SetKernelArgMem(0, v1)
                        .SetKernelArgMem(1, v2)
                        .SetKernelArgMem(2, result)
                        .SetKernelArg(3, sizeof(int), v2Index)
                        .EnqueueNDRangeKernel(
                            1
                        );

                    clProvider.QueueFinish();

                    result.Read(BlockModeEnum.Blocking);
                    
                    var openclResult = result.Array[0];

                    var diff = openclResult - csharpResult;

                    Console.WriteLine(
                        "C# = {0},  OpenCL = {1},  diff = {2}",
                        csharpResult.ToString(),
                        openclResult.ToString(),
                        diff.ToString()
                        );

                    Assert.IsTrue(Math.Abs(diff) < epsilon);
                }
            }
        }

        private const string CheckKernel = @"
{CALCULATE_METRIC_BODY}

__kernel void {KERNEL_NAME}(
    __global float* v1,
    __global float* v2,
    __global float* result,
    int v2Index
    )
{
    result[0] = {CALCULATE_METRIC_CALL}(v1, v2, v2Index);
}
";

    }
}