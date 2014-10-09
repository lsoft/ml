using System;
using MathNet.Numerics.Distributions;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.DropConnect.Inferencer.OpenCL.GPU
{
    /// <summary>
    /// GPU implementation of layer inferencer that calculate Gaussian random on C#
    /// Recommends to use it in cases of using discrete GPU hardware.
    /// </summary>
    public class GPULayerInferencer : ILayerInferencer
    {

        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly int _sampleCount;
        private readonly ILayer _previousLayer;
        private readonly ILayer _currentLayer;
        private readonly MemFloat _weightMem;
        private readonly MemFloat _previousLayerStateMem;
        private readonly MemFloat _currentLayerStateMem;
        private readonly float _p;
        private readonly int _randomCount;

        private MemFloat _randomMem;
        private Kernel _inferenceKernel;


        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="sampleCount">Sample count per neuron per inference iteration (typically 1000 - 10000)</param>
        /// <param name="previousLayer">Previous layer of dropconnect MLP (the algorithm needs to know neuron count of previous layer)</param>
        /// <param name="currentLayer">Current layer of dropconnect MLP (the algorithm needs to know neuron count and activation function of current layer)</param>
        /// <param name="weightMem">Weights of current MLP layer</param>
        /// <param name="previousLayerStateMem">State of previous layer neurons</param>
        /// <param name="currentLayerStateMem">State of current layer neurons</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public GPULayerInferencer(
            IRandomizer randomizer,
            CLProvider clProvider,
            int sampleCount,
            ILayer previousLayer,
            ILayer currentLayer,
            MemFloat weightMem,
            MemFloat previousLayerStateMem,
            MemFloat currentLayerStateMem,
            float p
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (previousLayer == null)
            {
                throw new ArgumentNullException("previousLayer");
            }
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }
            if (weightMem == null)
            {
                throw new ArgumentNullException("weightMem");
            }
            if (previousLayerStateMem == null)
            {
                throw new ArgumentNullException("previousLayerStateMem");
            }
            if (currentLayerStateMem == null)
            {
                throw new ArgumentNullException("currentLayerStateMem");
            }

            _randomizer = randomizer;
            _clProvider = clProvider;
            _sampleCount = sampleCount;
            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _weightMem = weightMem;
            _previousLayerStateMem = previousLayerStateMem;
            _currentLayerStateMem = currentLayerStateMem;
            _p = p;

            _randomCount = _sampleCount * 32;

            RegisterOpenCLComponents();
        }

        private void RegisterOpenCLComponents()
        {
            //создаем и заполняем хранилище рандомов
            _randomMem = _clProvider.CreateFloatMem(
                _randomCount,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            var normal = new Normal(0, 1);
            normal.RandomSource = new Random(_randomizer.Next(1000000));

            _randomMem.Array.Fill(() => (float)normal.Sample());

            _randomMem.Write(BlockModeEnum.NonBlocking);

            //создаем кернел выведения
            var activationFunction = _currentLayer.LayerActivationFunction.GetOpenCLActivationFunction("grnd");

            var kernelSource = InferenceKernelSource.Replace(
                "<activationFunction_grnd>",
                activationFunction);

            _inferenceKernel = _clProvider.CreateKernel(
                kernelSource,
                "InferenceKernel");
        }

        public void InferenceLayer()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            var startRandomIndex = _randomizer.Next(_randomCount);

            const int localSize = 256;

            _inferenceKernel
                .SetKernelArgMem(0, this._randomMem)
                .SetKernelArgMem(1, this._previousLayerStateMem)
                .SetKernelArgMem(2, this._weightMem)
                .SetKernelArgMem(3, this._currentLayerStateMem)
                .SetKernelArgLocalMem(4, localSize * sizeof(float))
                .SetKernelArg(5, 4, this._p)
                .SetKernelArg(6, 4, startRandomIndex)
                .SetKernelArg(7, 4, _randomCount)
                .SetKernelArg(8, 4, this._previousLayer.Neurons.Length)
                .SetKernelArg(9, 4, _sampleCount)
                .EnqueueNDRangeKernel(
                    new[]
                    {
                        _currentLayer.NonBiasNeuronCount * localSize
                    },
                    new[]
                    {
                        localSize
                    });

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
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
