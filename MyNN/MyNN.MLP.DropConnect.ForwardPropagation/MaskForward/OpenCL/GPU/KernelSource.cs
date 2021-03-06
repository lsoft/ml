﻿using System;
using System.Globalization;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU
{
    public class KernelSource
    {
        private const string ActivationMethodName = "Activate";

        public string GetKernelSource(
            IFunction function,
            int currentLayerTotalNeuronCount,
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
                currentLayerTotalNeuronCount.ToString(CultureInfo.InvariantCulture));

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
    const __global uint * mask,
    uint bitmask,
    __local float* partialDotProduct,
    __global float * biases

    )
{
    const uint width = {PREVIOUS_LAYER_NEURON_COUNT};
    const uint height = {CURRENT_LAYER_NEURON_COUNT};

   for (uint y = get_group_id(0); y < height; y += get_num_groups(0))
   {
      const __global float* row = weights + y * width;
      const __global uint* maskrow = mask + y * width;

      // Each work-item accumulates as many products as necessary
      // into private variable 'sum'
      float sum = 0;
      for (uint x = get_local_id(0); x < width; x += get_local_size(0))
      {
          uint maski = maskrow[x];
          float mask = ((maski & bitmask) > 0) ? (float)1 : (float)0;

          sum += row[x] * previousLayerLastState[x] * mask;
      }

      // Each partial dot product is stored in shared memory
      partialDotProduct[get_local_id(0)] = sum;

      barrier(CLK_LOCAL_MEM_FENCE);

      WarpReductionToFirstElement(partialDotProduct);

      // Write the result of the reduction to global memory
      if (get_local_id(0) == 0)
      {
         float lastNET = partialDotProduct[0] + biases[y];
         currentLayerLastNET[y] = lastNET;

         //compute last state
         float lastState = <ActivationMethodCall>(lastNET);
         currentLayerLastState[y] = lastState;
      }

      // Synchronize to make sure the first work-item is done
      barrier(CLK_LOCAL_MEM_FENCE);
   }
}
";
    }
}
