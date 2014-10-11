using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.ActivationFunction.OpenCL.Value
{
    internal class ActivationFunctionValueTests
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
                var diffk = CalculateKernel.Replace(
                    "<GetActivationFunction>",
                    f.GetOpenCLActivationMethod(
                        "GetActivationFunction", 
                        VectorizationSizeEnum.NoVectorization));

                var calculateKernel = clProvider.CreateKernel(
                    diffk,
                    "CalculateValuesKernel");

                var count = (int)((right - left) / step);

                //считаем через С#
                var values = new float[count];
                var currentcc = left;
                for (var cc = 0; cc < count; cc++)
                {
                    var value = f.Compute(currentcc);
                    values[cc] = value;

                    currentcc += step;
                }

                var valueMem = clProvider.CreateFloatMem(
                    count, 
                    MemFlags.CopyHostPtr | MemFlags.ReadWrite
                    );

                valueMem.Write(BlockModeEnum.Blocking);

                calculateKernel
                    .SetKernelArgMem(0, valueMem)
                    .SetKernelArg(1, 4, left)
                    .SetKernelArg(2, 4, step)
                    .EnqueueNDRangeKernel(count);

                clProvider.QueueFinish();

                valueMem.Read(BlockModeEnum.Blocking);

                var diffArray = ArrayOperations.DiffArrays(
                    values,
                    valueMem.Array
                    );

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Разница: {0}",
                        string.Join(
                            "   ",
                            diffArray)));

                for (var cc = 0; cc < diffArray.Length; cc++)
                {
                    var diff = diffArray[cc];

                    Assert.IsTrue(diff <= allowedDerivativeEpsilon);
                }

            }
        }


        private const string CalculateKernel = @"
<GetActivationFunction>

__kernel void CalculateValuesKernel(
    __global float * valueMem,
    const float minValue,
    const float step)
{
    int kernelIndex = get_global_id(0);

    float cc = minValue + kernelIndex * step;

    float value = GetActivationFunction(cc);

    valueMem[kernelIndex] = value;
}

";
    }
}