using System;
using System.Globalization;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Classic.ForwardPropagation.OpenCL.Img.GPU
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
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void ComputeLayerKernel(
    read_only  image2d_t previousLayerLastState,
    write_only image2d_t currentLayerLastNET,
    write_only image2d_t currentLayerLastState,
    read_only  image2d_t weights,
    __local float* partialDotProduct
    )
{
    uint width = {PREVIOUS_LAYER_NEURON_COUNT};
    uint height = {CURRENT_LAYER_NEURON_COUNT};

   for (uint y = get_group_id(0); y < height; y += get_num_groups(0))
   {
      // Each work-item accumulates as many products as necessary
      // into private variable 'sum'
      float sum = 0;
      for (uint x = get_local_id(0); x < width; x += get_local_size(0))
      {
           float rowx = read_imagef(weights, sampler, (int2)(x, y)).s0;
           float previousLayerLastStatex = read_imagef(previousLayerLastState, sampler, (int2)(0, x)).s0;

           sum += rowx * previousLayerLastStatex ;
      }

      // Each partial dot product is stored in shared memory
      partialDotProduct[get_local_id(0)] = sum;

      barrier(CLK_LOCAL_MEM_FENCE);

      WarpReductionToFirstElement(partialDotProduct);

      // Write the result of the reduction to global memory
      if (get_local_id(0) == 0)
      {
         float lastNET = partialDotProduct[0];
         write_imagef(currentLayerLastNET, (int2)(0, y), (float4)(lastNET, lastNET, lastNET, lastNET));

         //compute last state
         float lastState = <activationFunction_lastNET>;
         write_imagef(currentLayerLastState, (int2)(0, y), (float4)(lastState, lastState, lastState, lastState));
      }

      // Synchronize to make sure the first work-item is done with
      // reading partialDotProduct
      barrier(CLK_LOCAL_MEM_FENCE);
   }
}
";
    }
}
