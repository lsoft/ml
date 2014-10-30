using System;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.DropConnect.Inferencer.OpenCL.CPU.Vectorized
{
    public class KernelSource
    {
        private const string ActivationMethodName = "Activate";

        public string GetKernelSource(
            IFunction function,
            out string kernelName
            )
        {
            if (function == null)
            {
                throw new ArgumentNullException("function");
            }

            var result = InferenceKernelSource.Replace(
                "<ActivationMethodCall>",
                ActivationMethodName);

            var activationMethod = function.GetOpenCLActivationMethod(
                ActivationMethodName,
                VectorizationSizeEnum.VectorizationMode16
                );

            result = result.Replace(
                "<ActivationMethodBody>",
                activationMethod);

            kernelName = "InferenceKernel16";

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

<ActivationMethodBody>

__kernel void
        InferenceKernel16(
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

    int workStartRandomIndexD16 = workStartRandomIndex / 16;
    int ostatok = workStartRandomIndex % 16;

    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics




//    float16 lastStateSummator16 = 0;
//    for(int sampleIndex = workStartRandomIndexD16; sampleIndex < (workStartRandomIndexD16 + sampleCount / 16); sampleIndex++)
//    {
//        float16 ogrnd16 = vload16(sampleIndex, randomMem + ostatok);
//
//        float16 grnd16 = ogrnd16 * wv_sigma + wv_median;
//
//        //compute last state
//        float16 lastState16 = <ActivationMethodCall>(grnd16);
//
//        lastStateSummator16 += lastState16;
//    }
//
//    float lastStateSummator = 
//          lastStateSummator16.s0 
//        + lastStateSummator16.s1 
//        + lastStateSummator16.s2 
//        + lastStateSummator16.s3
//        + lastStateSummator16.s4
//        + lastStateSummator16.s5
//        + lastStateSummator16.s6
//        + lastStateSummator16.s7
//        + lastStateSummator16.s8
//        + lastStateSummator16.s9
//        + lastStateSummator16.sa
//        + lastStateSummator16.sb
//        + lastStateSummator16.sc
//        + lastStateSummator16.sd
//        + lastStateSummator16.se
//        + lastStateSummator16.sf
//        ;
//


    KahanAccumulator16 acc16 = GetEmptyKahanAcc16();

    for(int sampleIndex = workStartRandomIndexD16; sampleIndex < (workStartRandomIndexD16 + sampleCount / 16); sampleIndex++)
    {
        float16 ogrnd16 = vload16(sampleIndex, randomMem + ostatok);

        float16 grnd16 = ogrnd16 * wv_sigma + wv_median;

        //compute last state
        float16 lastState16 = <ActivationMethodCall>(grnd16);

        KahanAddElement16(&acc16, lastState16);
    }

    float lastStateSummator = ReduceAcc16(&acc16);




    //усредняем
    float result = lastStateSummator / sampleCount;

    //записываем обратно в хранилище
    currentLayerLastState[neuronIndex] = result;
}

";
    }
}