﻿using System;
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
        /// <param name="p">Probability for each bit to be ONE (TRUE) (with p = 1 it completely disables mask and convert the model to classic backprop)</param>
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
            var ks = new KernelSource();

            string kernelName;
            var kernelSource = ks.GetKernelSource(
                _currentLayer.LayerActivationFunction,
                out kernelName
                );

            _inferenceKernel = _clProvider.CreateKernel(
                kernelSource,
                kernelName
                );
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
                .SetKernelArg(8, 4, this._previousLayer.TotalNeuronCount)
                .SetKernelArg(9, 4, _sampleCount)
                .EnqueueNDRangeKernel(
                    new[]
                    {
                        _currentLayer.TotalNeuronCount * localSize
                    },
                    new[]
                    {
                        localSize
                    });

            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }
    }
}
