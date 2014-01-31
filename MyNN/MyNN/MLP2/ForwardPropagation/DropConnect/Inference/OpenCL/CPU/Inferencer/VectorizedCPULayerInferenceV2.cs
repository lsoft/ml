using System;
using MathNet.Numerics.Distributions;
using MyNN.MLP2.Structure;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU.Inferencer
{
    /// <summary>
    /// Vectorized implementation of layer inferencer that calculate Gaussian random on C#
    /// It is most efficient implementation in case of CPU-OpenCL.
    /// Recommends to use it in cases inability to obtain discrete GPU hardware.
    /// </summary>
    public class VectorizedCPULayerInferenceV2 : ILayerInference
    {

        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly int _sampleCount;
        private readonly MLPLayer _previousLayer;
        private readonly MLPLayer _currentLayer;
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
        public VectorizedCPULayerInferenceV2(
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
            _sampleCount = sampleCount - sampleCount % 16;
            _previousLayer = previousLayer;
            _currentLayer = currentLayer;
            _weightMem = weightMem;
            _previousLayerStateMem = previousLayerStateMem;
            _currentLayerStateMem = currentLayerStateMem;
            _p = p;

            _randomCount = (_sampleCount * 3) - (_sampleCount * 3) % 16;

            if (_sampleCount != sampleCount)
            {
                ConsoleAmbientContext.Console.WriteLine(
                    "Inferencer: Input sample count {0} has changed to {1} due to vectorization mode 16",
                    sampleCount,
                    _sampleCount);
            }

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
            var activationFunction = _currentLayer.LayerActivationFunction.GetOpenCLActivationFunction("grnd16");

            var kernelSource = _inferenceKernelSource.Replace(
                "<activationFunction_grnd16>",
                activationFunction);

            _inferenceKernel = _clProvider.CreateKernel(
                kernelSource,
                "InferenceKernel16");
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

    //float16 wv_sigma16 = (float16)wv_sigma;
    //float16 wv_median16 = (float16)wv_median;

    int workStartRandomIndexD16 = workStartRandomIndex / 16;
    int ostatok = workStartRandomIndex % 16;

    float16 lastStateSummator16 = 0;
    for(int sampleIndex = workStartRandomIndexD16; sampleIndex < (workStartRandomIndexD16 + sampleCount / 16); sampleIndex++)
    {
        float16 ogrnd16 = vload16(sampleIndex, randomMem + ostatok);

        float16 grnd16 = ogrnd16 * wv_sigma + wv_median;
        //float16 grnd16 = mad(ogrnd16, wv_sigma16, wv_median);

        //compute last state
        float16 lastState16 = <activationFunction_grnd16>;

        lastStateSummator16 += lastState16;
    }

    float lastStateSummator = 
          lastStateSummator16.s0 
        + lastStateSummator16.s1 
        + lastStateSummator16.s2 
        + lastStateSummator16.s3
        + lastStateSummator16.s4
        + lastStateSummator16.s5
        + lastStateSummator16.s6
        + lastStateSummator16.s7
        + lastStateSummator16.s8
        + lastStateSummator16.s9
        + lastStateSummator16.sa
        + lastStateSummator16.sb
        + lastStateSummator16.sc
        + lastStateSummator16.sd
        + lastStateSummator16.se
        + lastStateSummator16.sf
        ;

    //усредняем
    float result = lastStateSummator / sampleCount;

    //записываем обратно в хранилище
    currentLayerLastState[neuronIndex] = result;
}

";
    }
}
