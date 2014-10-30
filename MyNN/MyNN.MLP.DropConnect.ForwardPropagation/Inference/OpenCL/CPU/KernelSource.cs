using System;
using System.Globalization;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.DropConnect.ForwardPropagation.Inference.OpenCL.CPU
{
    public class KernelSource
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
    const __global float* weights
    )
{
    const uint width = {PREVIOUS_LAYER_NEURON_COUNT};
    const uint height = {CURRENT_LAYER_NEURON_COUNT};

    uint y = get_global_id(0);

    const __global float* row = weights + y * width;

    KahanAccumulator acc = GetEmptyKahanAcc();
    for (int x = 0; x < width; ++x)
    {
        float incre = row[x] * previousLayerLastState[x];

        KahanAddElement(&acc, incre);
    }

    float lastNET = acc.Sum;

    currentLayerLastNET[y] = lastNET;

    //compute last state
    float lastState = <ActivationMethodCall>(lastNET);
    currentLayerLastState[y] = lastState;
}
";
    }
}
