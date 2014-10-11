﻿using System;
using System.Globalization;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.CPU
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
    uint bitmask
    )
{
    const uint width = {PREVIOUS_LAYER_NEURON_COUNT};
    const uint height = {CURRENT_LAYER_NEURON_COUNT};

    uint y = get_global_id(0);

    const __global float* row = weights + y * width;
    const __global uint* maskrow = mask + y * width;

    KahanAccumulator acc = GetEmptyKahanAcc();
    for (int x = 0; x < width; ++x)
    {
        uint maski = maskrow[x];
        float mask = ((maski & bitmask) > 0) ? (float)1 : (float)0;

        float incre = row[x] * previousLayerLastState[x] * mask;

        KahanAddElement(&acc, incre);
    }

    float lastNET = acc.Sum;

    currentLayerLastNET[y] = lastNET;

    //compute last state
    float lastState = <activationFunction_lastNET>;
    currentLayerLastState[y] = lastState;
}
";
    }
}