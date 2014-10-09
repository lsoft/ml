using System;
using System.Globalization;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU
{
    public class KernelSource
    {
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

            var activationFunction = function.GetOpenCLActivationFunction("lastNET");

            var result = ComputeWeightSource;

            result += KernelSourceCode.Replace(
                "<activationFunction_lastNET>",
                activationFunction);

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
__kernel void ComputeLayerKernel(
    const __global float* previousLayerLastState,
    __global float* currentLayerLastNET,
    __global float * currentLayerLastState,
    const __global float* weights,
    const __global uint * mask,
    uint bitmask,
    __local float* partialDotProduct
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
         float lastNET = partialDotProduct[0];
         currentLayerLastNET[y] = lastNET;

         //compute last state
         float lastState = <activationFunction_lastNET>;
         currentLayerLastState[y] = lastState;
      }

      // Synchronize to make sure the first work-item is done
      barrier(CLK_LOCAL_MEM_FENCE);
   }
}
";
    }
}
