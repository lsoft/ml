using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Accord.Math;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using Normal = MathNet.Numerics.Distributions.Normal;

namespace MyNN.MLP2.ForwardPropagation.DropConnect
{
    public class OpenCLLayerInferenceMedian : ILayerInference
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

        private Kernel _inferenceKernel;


        public OpenCLLayerInferenceMedian(
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

            throw new InvalidOperationException(
                "This implementation of drop connect inference is incorrect. Imagine RLU function, where median is negative. This inferencer returns zero, but correct inferencer give us a value that more than zero (some piece of samples will be positive so RLU will be positive too).");
        }

        private void RegisterOpenCLComponents()
        {
            //создаем кернел выведения
            var activationFunction = _currentLayer.LayerActivationFunction.GetOpenCLActivationFunction("wv_median");

            var kernelSource = _inferenceKernelSource.Replace(
                "<activationFunction_wv_median>",
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
            0   __global float * previousLayerLastState,
            1   __global float * weights,
            2   __global float * currentLayerLastState,
            3   float p,
            4   int previousLayerNeuronCountTotal,
            5   int sampleCount)
            //*/

            _inferenceKernel
                .SetKernelArgMem(0, this._previousLayerStateMem)
                .SetKernelArgMem(1, this._weightMem)
                .SetKernelArgMem(2, this._currentLayerStateMem)
                .SetKernelArg(3, 4, this._p)
                .SetKernelArg(4, 4, this._previousLayer.Neurons.Length)
                .SetKernelArg(5, 4, _sampleCount)
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
            __global float * previousLayerLastState,
            __global float * weights,
            __global float * currentLayerLastState,
            float p,
            int previousLayerNeuronCountTotal,
            int sampleCount)
{
    int neuronIndex = get_global_id(0);

    //суммируем веса * состояние нейронов пред. слоя и высчитываем медиану и сигма-квадрат для гауссианы
    int weightIndex = ComputeWeightIndex(previousLayerNeuronCountTotal, neuronIndex);

    float wv_median  = 0;
    for (int plnIndex = 0; plnIndex < previousLayerNeuronCountTotal; ++plnIndex)
    {
        float wv = weights[weightIndex++] * previousLayerLastState[plnIndex];

        wv_median += wv;
    }

    wv_median *= p;

    float lastState = <activationFunction_wv_median>;

    //записываем обратно в хранилище
    currentLayerLastState[neuronIndex] = lastState;
}

";
    }
}
