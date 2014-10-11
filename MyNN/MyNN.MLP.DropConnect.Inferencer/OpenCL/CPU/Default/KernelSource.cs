using System;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.DropConnect.Inferencer.OpenCL.CPU.Default
{
    internal class KernelSource
    {
        public string GetKernelSource(
            IFunction function,
            out string kernelName
            )
        {
            if (function == null)
            {
                throw new ArgumentNullException("function");
            }

            var activationFunction = function.GetOpenCLActivationFunction("grnd");

            var result = InferenceKernelSource.Replace(
                "<activationFunction_grnd>",
                activationFunction);

            kernelName = "InferenceKernel1";

            return
                result;
        }

        private const string InferenceKernelSource = @"
inline int ComputeWeightIndex(
    int previousLayerNeuronCount,
    int neuronIndex)
{
    return
        previousLayerNeuronCount * neuronIndex;
}

__kernel void
        InferenceKernel1(
            __global float * randomMem,
            __global float * previousLayerLastState,
            __global float * weights,
            __global float * currentLayerLastState,
            float p,
            int startRandomIndex,
            int randomSize,
            int previousLayerNeuronCountTotal,
            int sampleCount)
{
    int neuronIndex = get_global_id(0);

    //суммируем веса * состояние нейронов пред. слоя и высчитываем медиану и сигма-квадрат для гауссианы
    int weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);


    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics



//    float wv_median  = 0;
//    float wv_sigmasq = 0;
//    for (int plnIndex = 0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
//    {
//        float wv = weights[weightIndex++] * previousLayerLastState[plnIndex];
//
//        wv_median += wv;
//        wv_sigmasq += wv * wv;
//    }




    KahanAccumulator accMedian = GetEmptyKahanAcc();
    KahanAccumulator accSigmaSq = GetEmptyKahanAcc();

    for (int plnIndex = 0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
    {
        float wv = weights[weightIndex++] * previousLayerLastState[plnIndex];

        KahanAddElement(&accMedian, wv);
        KahanAddElement(&accSigmaSq, wv * wv);
    }

    float wv_median  = accMedian.Sum;
    float wv_sigmasq = accSigmaSq.Sum;




    wv_median *= p;
    wv_sigmasq *= p * (1 - p);

    float wv_sigma = sqrt(wv_sigmasq);

    //начинаем семплировать из гауссианы и гнать через функцию активации
    int workStartRandomIndex = (startRandomIndex + neuronIndex * previousLayerNeuronCountTotal) % randomSize;

    if((workStartRandomIndex + sampleCount) >= randomSize)
    {
        if(workStartRandomIndex > sampleCount)
        {
            workStartRandomIndex -= sampleCount;
        }
        else
        {
            workStartRandomIndex = 0;
        }
    }

    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics

//    float lastStateSummator  = 0;
//    for(int sampleIndex = workStartRandomIndex; sampleIndex < (workStartRandomIndex + sampleCount); sampleIndex++)
//    {
//        //делаем гауссиану с медианой wv_median и сигмой wv_sigma из гауссианы (0;1), пришедшей из C#
//        float ogrnd = randomMem[sampleIndex];
//        float grnd = ogrnd * wv_sigma + wv_median;
//
//        //compute last state
//        float lastState = <activationFunction_grnd>;
//
//        lastStateSummator += lastState;
//    }


    KahanAccumulator accState = GetEmptyKahanAcc();

    for(int sampleIndex = workStartRandomIndex; sampleIndex < (workStartRandomIndex + sampleCount); sampleIndex++)
    {
        //делаем гауссиану с медианой wv_median и сигмой wv_sigma из гауссианы (0;1), пришедшей из C#
        float ogrnd = randomMem[sampleIndex];
        float grnd = ogrnd * wv_sigma + wv_median;

        //compute last state
        float lastState = <activationFunction_grnd>;

        KahanAddElement(&accState, lastState);
    }

    float lastStateSummator  = accState.Sum;

    //усредняем
    float result = lastStateSummator / sampleCount;

    //записываем обратно в хранилище
    currentLayerLastState[neuronIndex] = result;
}

";
    }
}