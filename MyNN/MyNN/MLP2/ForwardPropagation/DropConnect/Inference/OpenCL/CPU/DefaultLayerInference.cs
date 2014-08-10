using System;
using MathNet.Numerics.Distributions;
using MyNN.MLP2.Structure.Layer;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU
{
    /// <summary>
    /// Implementation of layer inferencer that calculate Gaussian random on C#
    /// It is more efficient implementation than NaiveLayerInference and may be used as starting point of GPU-version
    /// </summary>
    public class DefaultLayerInference : ILayerInference
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
        public DefaultLayerInference(
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

            _randomCount = (sampleCount * 3) - (sampleCount * 3) % 16;

            RegisterOpenCLComponents();
        }

        private void RegisterOpenCLComponents()
        {
            //создаем и заполняем хранилище рандомов
            _randomMem = _clProvider.CreateFloatMem(
                _randomCount,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            var normal = new Normal(0, 1);
            normal.RandomSource = new Random(_randomizer.Next(100000));
            
            for (var cc = 0; cc < _randomCount; cc++)
            {
                _randomMem.Array[cc] = (float)normal.Sample();
            }

            _randomMem.Write(BlockModeEnum.NonBlocking);

            //создаем кернел выведения
            var activationFunction = _currentLayer.LayerActivationFunction.GetOpenCLActivationFunction("grnd");

            var kernelSource = _inferenceKernelSource.Replace(
                "<activationFunction_grnd>",
                activationFunction);

            _inferenceKernel = _clProvider.CreateKernel(
                kernelSource,
                "InferenceKernel1");
        }

        public void InferenceLayer()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();

            /*
            0   __global float * randomMem,
            1   __global float * previousLayerLastState,
            2   __global float * weights,
            3   __global float * currentLayerLastState,
            4   float p,
            5   int startRandomIndex,
            6   int randomSize,
            7   int previousLayerNeuronCountTotal,
            8   int sampleCount)
            //*/

            var startRandomIndex = _randomizer.Next(_randomCount);

            _inferenceKernel
                .SetKernelArgMem(0, this._randomMem)
                .SetKernelArgMem(1, this._previousLayerStateMem)
                .SetKernelArgMem(2, this._weightMem)
                .SetKernelArgMem(3, this._currentLayerStateMem)
                .SetKernelArg(4, 4, this._p)
                .SetKernelArg(5, 4, startRandomIndex)
                .SetKernelArg(6, 4, _randomCount)
                .SetKernelArg(7, 4, this._previousLayer.Neurons.Length)
                .SetKernelArg(8, 4, _sampleCount)
                .EnqueueNDRangeKernel(_currentLayer.NonBiasNeuronCount);

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }


        private const string _inferenceKernelSource = @"
int ComputeWeightIndex(
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

    float wv_median  = 0;
    float wv_sigmasq = 0;
    for (int plnIndex = 0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
    {
        float wv = weights[weightIndex++] * previousLayerLastState[plnIndex];

        wv_median += wv;
        wv_sigmasq += wv * wv;
    }

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


    float lastStateSummator  = 0;
    for(int sampleIndex = workStartRandomIndex; sampleIndex < (workStartRandomIndex + sampleCount); sampleIndex++)
    {
        //делаем гауссиану с медианой wv_median и сигмой wv_sigma из гауссианы (0;1), пришедшей из C#
        float ogrnd = randomMem[sampleIndex];
        float grnd = ogrnd * wv_sigma + wv_median;

        //compute last state
        float lastState = <activationFunction_grnd>;

        lastStateSummator += lastState;
    }

    //усредняем
    float result = lastStateSummator / sampleCount;

    //записываем обратно в хранилище
    currentLayerLastState[neuronIndex] = result;
}

";
    }
}
