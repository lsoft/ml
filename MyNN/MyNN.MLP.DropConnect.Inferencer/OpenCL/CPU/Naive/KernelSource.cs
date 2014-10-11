using System;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.DropConnect.Inferencer.OpenCL.CPU.Naive
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
//Box-Muller
float SampleFromGaussian(float rnd1, float rnd2, float median, float sigma)
{
    float f = sqrt(-2 * log(rnd1)) * cos(2 * M_PI_F * rnd2);

    float r = f * sigma + median;

    return r;
}

float SampleFromGaussian2(
    __global float * randomMem,
    int * randomIndex,
    int randomSize,
    float median,
    float sigma)
{
    int index = *randomIndex;

    float rnd1 = randomMem[index];

    index = (index + 1) % randomSize;

    float rnd2 = randomMem[index];

    index = (index + 1) % randomSize;

    *randomIndex = index;

    return
        SampleFromGaussian(rnd1, rnd2, median, sigma);
}

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

    //начинаем семплировать из гауссианы и гнать через функцию активации
    int workStartRandomIndex = (startRandomIndex + neuronIndex * previousLayerNeuronCountTotal) % randomSize;
    
    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics


//    float lastStateSummator  = 0;
//    for(int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
//    {
//        float grnd = SampleFromGaussian2(
//            randomMem,
//            &workStartRandomIndex,
//            randomSize,
//            wv_median,
//            sqrt(wv_sigmasq));
//
//        //compute last state
//        float lastState = <activationFunction_grnd>;
//
//        lastStateSummator += lastState;
//    }



    KahanAccumulator accState = GetEmptyKahanAcc();

    for(int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        float grnd = SampleFromGaussian2(
            randomMem,
            &workStartRandomIndex,
            randomSize,
            wv_median,
            sqrt(wv_sigmasq));

        //compute last state
        float lastState = <activationFunction_grnd>;

        KahanAddElement(&accState, lastState);
    }

    float lastStateSummator  = accState.Sum;



    //усредняем
    float result = lastStateSummator / sampleCount;
    
    currentLayerLastState[neuronIndex] = result;
}

";
    }
}