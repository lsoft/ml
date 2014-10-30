using System;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Classic.ForwardPropagation.OpenCL.Mem.CPU
{
    public class CPUKernelSource
    {
        private const string ActivationMethodName = "Activate";

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

            var result = ComputeWeightSource;

            result += kernelSource.Replace(
                "<ActivationMethodCall>",
                ActivationMethodName);

            var activationMethod = function.GetOpenCLActivationMethod(
                ActivationMethodName,
                VectorizationSizeEnum.NoVectorization
                );

            result = result.Replace(
                "<ActivationMethodBody>",
                activationMethod);

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

        private const string KernelSource1 = @"
<ActivationMethodBody>

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

    int weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

    //compute LastNET
    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics

    KahanAccumulator acc = GetEmptyKahanAcc();

    for (int plnIndex =0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
    {
        float lastNETIncrement = weights[weightIndex++] * previousLayerLastState[plnIndex];

        KahanAddElement(&acc, lastNETIncrement);
    }

    float lastNET = acc.Sum;



    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <ActivationMethodCall>(lastNET);
    currentLayerLastState[neuronIndex] = lastState;
}
";

        private const string KernelSource4 = @"
<ActivationMethodBody>

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

    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics

    KahanAccumulator4 acc4 = GetEmptyKahanAcc4();
    
    for (
        int plnIndex4 = 0, //индексатор внутри состояния нейронов пред. слоя
            weightIndex4 = beginWeightIndex / 4, //индексатор на первый элемент float4 в массиве весов
            weightShift4 = beginWeightIndex - weightIndex4 * 4; //смещение для получения правильного float4 (так как например, для нейрона 33 и нейронов пред. слоя 127 смещение будет 4191, не кратно 4)
        plnIndex4 < previousLayerNeuronCount4;
        ++plnIndex4, ++weightIndex4)
    {
        float4 weights4 = vload4(weightIndex4, weights + weightShift4);
        float4 previousLayerLastState4 = vload4(plnIndex4, previousLayerLastState);

        KahanAddElement4(&acc4, weights4 * previousLayerLastState4);
    }


    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics
    //добираем невекторизованные данные (максимум - 3 флоата)

    KahanAccumulator acc = GetKahanAcc(ReduceAcc4(&acc4));

    //добираем невекторизованные данные (максимум - 3 флоата)
    for (
        int plnIndex = previousLayerNeuronCount4M4, //индексатор внутри состояния нейронов пред. слоя
            weightIndex = beginWeightIndex + previousLayerNeuronCount4M4; //индексатор на массив весов
        plnIndex < previousLayerNeuronCountTotal;
        ++plnIndex, ++weightIndex)
    {
        float lastNETIncrement = weights[weightIndex] * previousLayerLastState[plnIndex];

        KahanAddElement(&acc, lastNETIncrement);
    }

    float lastNET = acc.Sum;



    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <ActivationMethodCall>(lastNET);
    currentLayerLastState[neuronIndex] = lastState;
}
";

        private const string KernelSource16 = @"
<ActivationMethodBody>

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

    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics

    KahanAccumulator16 acc16 = GetEmptyKahanAcc16();

    for (
        int plnIndex16 = 0, //индексатор внутри состояния нейронов пред. слоя
            weightIndex16 = beginWeightIndex / 16, //индексатор на первый элемент float16 в массиве весов
            weightShift16 = beginWeightIndex - weightIndex16 * 16; //смещение для получения правильного float16 (так как например, для нейрона 33 и нейронов пред. слоя 127 смещение будет 4191, не кратно 4)
        plnIndex16 < previousLayerNeuronCount16;
        ++plnIndex16, ++weightIndex16)
    {
        float16 weights16 = vload16(weightIndex16, weights + weightShift16);
        float16 previousLayerLastState16 = vload16(plnIndex16, previousLayerLastState);

        KahanAddElement16(&acc16, weights16 * previousLayerLastState16);
    }

    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics
    //добираем невекторизованные данные (максимум - 15 флоатов)

    KahanAccumulator acc = GetKahanAcc(ReduceAcc16(&acc16));

    //добираем невекторизованные данные (максимум - 15 флоатов)
    for (
        int plnIndex = previousLayerNeuronCount16M16, //индексатор внутри состояния нейронов пред. слоя
            weightIndex = beginWeightIndex + previousLayerNeuronCount16M16; //индексатор на массив весов
        plnIndex < previousLayerNeuronCountTotal;
        ++plnIndex, ++weightIndex)
    {
        float lastNETIncrement = weights[weightIndex] * previousLayerLastState[plnIndex];

        KahanAddElement(&acc, lastNETIncrement);
    }

    float lastNET = acc.Sum;



    currentLayerLastNET[neuronIndex] = lastNET;

    //compute last state

    float lastState = <ActivationMethodCall>(lastNET);
    currentLayerLastState[neuronIndex] = lastState;
}
";
    }
}
