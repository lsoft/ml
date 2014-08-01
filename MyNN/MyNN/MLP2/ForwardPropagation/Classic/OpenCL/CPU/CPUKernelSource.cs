using System;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU
{
    public class CPUKernelSource
    {
        public string GetKernelSource(
            VectorizationSizeEnum vse,
            IFunction function,
            out string kernelName
            )
        {
            if (function == null)
            {
                throw new ArgumentNullException("function");
            }

            string kernelSource;
            switch (vse)
            {
                case VectorizationSizeEnum.NoVectorization:
                    kernelSource = KernelSource1;
                    break;
                case VectorizationSizeEnum.VectorizationMode4:
                    kernelSource = KernelSource4;
                    break;
                case VectorizationSizeEnum.VectorizationMode16:
                    kernelSource = KernelSource16;
                    break;
                default:
                    throw new ArgumentOutOfRangeException("vse");
            }

            var activationFunction = function.GetOpenCLActivationFunction("lastNET");

            var result = ComputeWeightSource;

            result += kernelSource.Replace(
                "<activationFunction_lastNET>",
                activationFunction);

            kernelName = "ComputeLayerKernel";

            return
                result;
        }

        private const string ComputeWeightSource = @"
int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}
";

        private const string KernelSource1 = @"
__kernel void
        ComputeLayerKernel(
            __global float * previousLayerLastState,
            __global float * currentLayerLastNET,
            __global float * currentLayerLastState,
            __global float * weights,
            int previousLayerNeuronCountTotal)
{
    //оригинальный алгоритм более чем в два раза медленен

    int neuronIndex = get_global_id(0);

    //compute LastNET
    int weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);
    float lastNET = 0;
    for (int plnIndex =0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
    {
        lastNET += weights[weightIndex++] * previousLayerLastState[plnIndex];
    }

    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <activationFunction_lastNET>;
    currentLayerLastState[neuronIndex] = lastState;
}
";

        private const string KernelSource4 = @"
__kernel void
        ComputeLayerKernel(
            __global float * previousLayerLastState,
            __global float * currentLayerLastNET,
            __global float * currentLayerLastState,
            __global float * weights,
            int previousLayerNeuronCountTotal)
{
    int previousLayerNeuronCount4 = previousLayerNeuronCountTotal / 4;
    int previousLayerNeuronCount4M4 = previousLayerNeuronCountTotal - previousLayerNeuronCountTotal % 4;

    int neuronIndex = get_global_id(0);

    //compute LastNET

    //забираем векторизованные данные

    //смещение в массиве весов на первый элемент
    int beginWeightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

    float4 lastNET4 = 0;
    for (
        int plnIndex4 = 0, //индексатор внутри состояния нейронов пред. слоя
            weightIndex4 = beginWeightIndex / 4, //индексатор на первый элемент float4 в массиве весов
            weightShift4 = beginWeightIndex - weightIndex4 * 4; //смещение для получения правильного float4 (так как например, для нейрона 33 и нейронов пред. слоя 127 смещение будет 4191, не кратно 4)
        plnIndex4 < previousLayerNeuronCount4;
        ++plnIndex4, ++weightIndex4)
    {
        float4 weights4 = vload4(weightIndex4, weights + weightShift4);
        float4 previousLayerLastState4 = vload4(plnIndex4, previousLayerLastState);

        lastNET4 += weights4 * previousLayerLastState4;
    }

    float lastNET = lastNET4.s0 + lastNET4.s1 + lastNET4.s2 + lastNET4.s3;

    //добираем невекторизованные данные (максимум - 3 флоата)
    for (
        int plnIndex = previousLayerNeuronCount4M4, //индексатор внутри состояния нейронов пред. слоя
            weightIndex = beginWeightIndex + previousLayerNeuronCount4M4; //индексатор на массив весов
        plnIndex < previousLayerNeuronCountTotal;
        ++plnIndex, ++weightIndex)
    {
        lastNET += weights[weightIndex] * previousLayerLastState[plnIndex];
    }

    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <activationFunction_lastNET>;
    currentLayerLastState[neuronIndex] = lastState;
}
";

        private const string KernelSource16 = @"
__kernel void
        ComputeLayerKernel(
            __global float * previousLayerLastState,
            __global float * currentLayerLastNET,
            __global float * currentLayerLastState,
            __global float * weights,
            int previousLayerNeuronCountTotal)
{
    int previousLayerNeuronCount16 = previousLayerNeuronCountTotal / 16;
    int previousLayerNeuronCount16M16 = previousLayerNeuronCountTotal - previousLayerNeuronCountTotal % 16;

    int neuronIndex = get_global_id(0);

    //compute LastNET

    //забираем векторизованные данные

    //смещение в массиве весов на первый элемент
    int beginWeightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

    float16 lastNET16 = 0;
    for (
        int plnIndex16 = 0, //индексатор внутри состояния нейронов пред. слоя
            weightIndex16 = beginWeightIndex / 16, //индексатор на первый элемент float16 в массиве весов
            weightShift16 = beginWeightIndex - weightIndex16 * 16; //смещение для получения правильного float16 (так как например, для нейрона 33 и нейронов пред. слоя 127 смещение будет 4191, не кратно 4)
        plnIndex16 < previousLayerNeuronCount16;
        ++plnIndex16, ++weightIndex16)
    {
        float16 weights16 = vload16(weightIndex16, weights + weightShift16);
        float16 previousLayerLastState16 = vload16(plnIndex16, previousLayerLastState);

        lastNET16 += weights16 * previousLayerLastState16;
    }

    float lastNET = 
          lastNET16.s0 
        + lastNET16.s1 
        + lastNET16.s2 
        + lastNET16.s3
        + lastNET16.s4
        + lastNET16.s5
        + lastNET16.s6
        + lastNET16.s7
        + lastNET16.s8
        + lastNET16.s9
        + lastNET16.sa
        + lastNET16.sb
        + lastNET16.sc
        + lastNET16.sd
        + lastNET16.se
        + lastNET16.sf
        ;

    //добираем невекторизованные данные (максимум - 15 флоатов)
    for (
        int plnIndex = previousLayerNeuronCount16M16, //индексатор внутри состояния нейронов пред. слоя
            weightIndex = beginWeightIndex + previousLayerNeuronCount16M16; //индексатор на массив весов
        plnIndex < previousLayerNeuronCountTotal;
        ++plnIndex, ++weightIndex)
    {
        lastNET += weights[weightIndex] * previousLayerLastState[plnIndex];
    }

    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <activationFunction_lastNET>;
    currentLayerLastState[neuronIndex] = lastState;
}
";
    }
}
