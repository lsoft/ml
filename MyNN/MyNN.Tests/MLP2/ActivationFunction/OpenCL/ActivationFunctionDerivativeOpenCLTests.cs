using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.ActivationFunction.OpenCL
{
    internal class ActivationFunctionDerivativeOpenCLTests
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
                var diffk0 = DiffKernel.Replace("<activationFunction_leftValue>", f.GetOpenCLActivationFunction("leftValue"));
                var diffk1 = diffk0.Replace("<activationFunction_rightValue>", f.GetOpenCLActivationFunction("rightValue"));
                var diffk2 = diffk1.Replace("<activationFunction_derivative_cc>", f.GetOpenCLFirstDerivative("cc"));

                var diffKernel = clProvider.CreateKernel(
                    diffk2,
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


                for (var cc = 0; cc < diffMem.Array.Length; cc++)
                {
                    var diff = diffMem.Array[cc];

                    Assert.IsTrue(diff <= allowedDerivativeEpsilon);
                }

            }
        }


        private const string DiffKernel = @"
__kernel void CalculateDiff(
    __global float * diffMem,
    float minValue,
    float step)
{
    float DeltaX = 0.01;

    int kernelIndex = get_global_id(0);
    float cc = minValue + kernelIndex * step;

    float leftValue = cc - DeltaX;
    float left =
        //f.Compute(leftValue);
        <activationFunction_leftValue>;

    float rightValue = cc + DeltaX;
    float right = 
        //f.Compute(rightValue);
        <activationFunction_rightValue>;

    float cDerivative = (right - left) / (2.0 * DeltaX);
    float fDerivative = 
        //f.ComputeFirstDerivative(cc);
        <activationFunction_derivative_cc>;

    float diff = fabs(cDerivative - fDerivative);

    diffMem[kernelIndex] = diff;
}
";
    }
}