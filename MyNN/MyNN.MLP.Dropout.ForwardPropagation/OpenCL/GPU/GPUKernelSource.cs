﻿using System;
using System.Globalization;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.GPU
{
    public class GPUKernelSource
    {
        private const string ActivationMethodName = "Activate";

        public string GetKernelSource(
            IFunction function,
            int currentLayerNonBiasNeuronCount,
            int previousLayerTotalNeuronCount,
            out string kernelName
            )
        {
            if (function == null)
            {
                throw new ArgumentNullException("function");
            }

            var result = ComputeWeightSource;

            result += KernelSourceCode.Replace(
                "<ActivationMethodCall>",
                ActivationMethodName);

            var activationMethod = function.GetOpenCLActivationMethod(
                ActivationMethodName,
                VectorizationSizeEnum.NoVectorization
                );

            result = result.Replace(
                "<ActivationMethodBody>",
                activationMethod);

            result = result.Replace(
                "{CURRENT_LAYER_NEURON_COUNT}",
                currentLayerNonBiasNeuronCount.ToString(CultureInfo.InvariantCulture));

            result = result.Replace(
                "{PREVIOUS_LAYER_NEURON_COUNT}",
                previousLayerTotalNeuronCount.ToString(CultureInfo.InvariantCulture));

            kernelName = "ComputeLayerKernel";

            return
                result;
        }

        private const string ComputeWeightSource = @"
inline int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}
";

        private const string KernelSourceCode = @"
<ActivationMethodBody>

__kernel void ComputeLayerKernel(
    const __global float* previousLayerLastState,
    __global float* currentLayerLastNET,
    __global float * currentLayerLastState,
    const __global float* weights,
    const __global uint * masks,
    int maskShift,
    uint bitmask,
    const float zeroValue0,
    const float oneValue0,
    const float zeroValue1,
    const float oneValue1,
    __local float* partialDotProduct,
    __global float * biases

    )
{
    const uint width = {PREVIOUS_LAYER_NEURON_COUNT};
    const uint height = {CURRENT_LAYER_NEURON_COUNT};

   for (uint y = get_group_id(0); y < height; y += get_num_groups(0))
   {
      const __global float* row = weights + y * width;

      // Each work-item accumulates as many products as necessary
      // into private variable 'sum'
      float sum = 0;
      for (uint x = get_local_id(0); x < width; x += get_local_size(0))
      {
           sum += row[x] * previousLayerLastState[x];
      }

      // Each partial dot product is stored in shared memory
      partialDotProduct[get_local_id(0)] = sum;

      barrier(CLK_LOCAL_MEM_FENCE);

      WarpReductionToFirstElement(partialDotProduct);

      // Write the result of the reduction to global memory
      if (get_local_id(0) == 0)
      {
         //вычисляем маску
         uint maski = masks[y + maskShift];
         float mask0 = ((maski & bitmask) > 0) ? oneValue0 : zeroValue0;
         float mask1 = ((maski & bitmask) > 0) ? oneValue1 : zeroValue1;

         //применяем маску
         float lastNET = (partialDotProduct[0] + biases[y]) * mask0;
         currentLayerLastNET[y] = lastNET;

         //compute last state
         float lastState = <ActivationMethodCall>(lastNET);
         currentLayerLastState[y] = lastState  * mask1;
      }

      // Synchronize to make sure the first work-item is done with
      // reading partialDotProduct
      barrier(CLK_LOCAL_MEM_FENCE);
   }
}
";
    }
}
