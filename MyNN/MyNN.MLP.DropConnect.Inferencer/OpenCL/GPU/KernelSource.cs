using System;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.DropConnect.Inferencer.OpenCL.GPU
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

            kernelName = "InferenceKernel";

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
        InferenceKernel(
            const __global float * randomMem,
            const __global float * previousLayerLastState,
            const __global float * weights,
            __global float * currentLayerLastState,
            __local float * local_results,
            float p,
            int startRandomIndex,
            int randomSize,
            int previousLayerNeuronCountTotal,
            int sampleCount)
{
    int neuronIndex = get_group_id(0);

    //суммируем веса * состояние нейронов пред. слоя и высчитываем медиану и сигма-квадрат для гауссианы
    //instead of plain summation we use Kahan algorithm due to more precision in floating point ariphmetics

    KahanAccumulator accMedian = GetEmptyKahanAcc();
    KahanAccumulator accSigmaSq = GetEmptyKahanAcc();

    int plnIndex = get_local_id(0);
    int weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex) +  get_local_id(0);
    for (; plnIndex < previousLayerNeuronCountTotal; weightIndex += get_local_size(0), plnIndex += get_local_size(0))
    {
        float wv = weights[weightIndex] * previousLayerLastState[plnIndex];

        KahanAddElement(&accMedian, wv);
        KahanAddElement(&accSigmaSq, wv * wv);
    }


    local_results[get_local_id(0)] = accMedian.Sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    WarpReductionToFirstElement(local_results);
    barrier(CLK_LOCAL_MEM_FENCE);
    float wv_median = local_results[0];

    //барьер ниже нужен, так как воркитем с локал_ид > 0 может еще не выполниться,
    //воркитем с локал_ид = 0 может уже перезатереть local_results[0] значением из 
    //accSigmaSq.Sum, что вызовет попадание accSigmaSq.Sum как wv_median для
    //воркитема с локал_ид > 0, что некорректно
    //особенно часто такое происходит на реализации Intel GPU OpenCL
    barrier(CLK_LOCAL_MEM_FENCE);

    local_results[get_local_id(0)] = accSigmaSq.Sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    WarpReductionToFirstElement(local_results);
    barrier(CLK_LOCAL_MEM_FENCE);
    float wv_sigmasq = local_results[0];


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

    KahanAccumulator acc = GetEmptyKahanAcc();

    for(int sampleIndex = workStartRandomIndex + get_local_id(0); sampleIndex < (workStartRandomIndex + sampleCount); sampleIndex += get_local_size(0))
    {
        float ogrnd = randomMem[sampleIndex];

        float grnd = ogrnd * wv_sigma + wv_median;

        //compute last state
        float lastState = <activationFunction_grnd>;

        KahanAddElement(&acc, lastState);
    }


    local_results[get_local_id(0)] = acc.Sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    WarpReductionToFirstElement(local_results);
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0)
    {
        float lastStateSummator = local_results[0];

        //усредняем
        float result = lastStateSummator / sampleCount;

        //записываем обратно в хранилище
        currentLayerLastState[neuronIndex] = result;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

}

";
    }
}