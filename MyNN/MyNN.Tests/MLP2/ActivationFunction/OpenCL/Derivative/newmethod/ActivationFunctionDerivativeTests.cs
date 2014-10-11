using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.ActivationFunction.OpenCL.Derivative.newmethod
{
    internal class ActivationFunctionDerivativeTests
    {
        public void ExecuteTests(
            IFunction f,
            float left = -100.5f,
            float right = 100f,
            float step = 0.17f,
            float allowedDerivativeEpsilon = 0.001f
            )
        {
            if (f == null)
            {
                throw new ArgumentNullException("f");
            }

            using (var clProvider = new CLProvider())
            {
                var diffk = DiffKernel.Replace(
                    "<GetActivationFunction>",
                    f.GetOpenCLActivationMethod(
                        "GetActivationFunction", 
                        VectorizationSizeEnum.NoVectorization));

                var diffKernel = clProvider.CreateKernel(
                    diffk,
                    "CalculateDiff");

                var count = (int)((right - left) / step);

                var diffMem = clProvider.CreateFloatMem(count, MemFlags.CopyHostPtr | MemFlags.ReadWrite);
                diffMem.Write(BlockModeEnum.Blocking);


                diffKernel
                    .SetKernelArgMem(0, diffMem)
                    .SetKernelArg(1, 4, left)
                    .SetKernelArg(2, 4, step)
                    .EnqueueNDRangeKernel(count);

                clProvider.QueueFinish();

                diffMem.Read(BlockModeEnum.Blocking);

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Разница: {0}",
                        string.Join(
                            "   ",
                            diffMem.Array)));

                for (var cc = 0; cc < diffMem.Array.Length; cc++)
                {
                    var diff = diffMem.Array[cc];

                    Assert.IsTrue(diff <= allowedDerivativeEpsilon);
                }

            }
        }


        private const string DiffKernel = @"
<GetActivationFunction>

__kernel void CalculateDiff(
    __global float * diffMem,
    const float minValue,
    const float step)
{

    float DeltaX = 0.01;

    int kernelIndex = get_global_id(0);
    float cc = minValue + kernelIndex * step;

    float leftValue = cc - DeltaX;
    float left = GetActivationFunction(leftValue);

    float rightValue = cc + DeltaX;
    float right = GetActivationFunction(rightValue);

    float cDerivative = (right - left) / (2.0 * DeltaX);
    float fDerivative = GetActivationFunction(cc);

    float diff = fabs(cDerivative - fDerivative);

    diffMem[kernelIndex] = diff;
}

";
    }
}