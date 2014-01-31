using System;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU.Inferencer
{
    /// <summary>
    /// Naive implementation of layer inferencer that calculate Gaussian random on OpenCL
    /// It is inefficient implementation and may be considered as OBSOLETE
    /// </summary>
    public class NaiveLayerInference : ILayerInference
    {
        private const int RandomCount = 131072;

        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly int _sampleCount;
        private readonly MLPLayer _previousLayer;
        private readonly MLPLayer _currentLayer;
        private readonly MemFloat _weightMem;
        private readonly MemFloat _previousLayerStateMem;
        private readonly MemFloat _currentLayerStateMem;
        private readonly float _p;

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
        public NaiveLayerInference(
            IRandomizer randomizer,
            CLProvider clProvider,
            int sampleCount,
            MLPLayer previousLayer,
            MLPLayer currentLayer,
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

            RegisterOpenCLComponents();
        }

        private void RegisterOpenCLComponents()
        {
            //создаем и заполняем хранилище рандомов
            _randomMem = _clProvider.CreateFloatMem(
                RandomCount,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            for (var cc = 0; cc < RandomCount; cc++)
            {
                _randomMem.Array[cc] = _randomizer.Next();
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

            var startRandomIndex = _randomizer.Next(RandomCount);

            _inferenceKernel
                .SetKernelArgMem(0, this._randomMem)
                .SetKernelArgMem(1, this._previousLayerStateMem)
                .SetKernelArgMem(2, this._weightMem)
                .SetKernelArgMem(3, this._currentLayerStateMem)
                .SetKernelArg(4, 4, this._p)
                .SetKernelArg(5, 4, startRandomIndex)
                .SetKernelArg(6, 4, RandomCount)
                .SetKernelArg(7, 4, this._previousLayer.Neurons.Length)
                .SetKernelArg(8, 4, _sampleCount)
                .EnqueueNDRangeKernel(_currentLayer.NonBiasNeuronCount);

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }


        private const string _inferenceKernelSource = @"
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

    //начинаем семплировать из гауссианы и гнать через функцию активации
    int workStartRandomIndex = (startRandomIndex + neuronIndex * previousLayerNeuronCountTotal) % randomSize;
    float lastStateSummator  = 0;
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

        lastStateSummator += lastState;
    }

    //усредняем
    float result = lastStateSummator / sampleCount;
    
    currentLayerLastState[neuronIndex] = result;
}

";
    }
}
