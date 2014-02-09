﻿using System;
using System.Collections.Generic;
using System.Threading;
using MathNet.Numerics.Distributions;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Float.WeightMask
{
    /// <summary>
    /// Implementation of weight float mask container for dropconnect backpropagation algorithm.
    /// This container uses mask array with size that greather than weights count.
    /// </summary>
    public class BigArrayWeightMaskContainer : IOpenCLWeightMaskContainer
    {
        /// <summary>
        /// Iteration count between random array recalculcation
        /// </summary>
        private const int TotalIterationCount = 100000;

        private readonly CLProvider _clProvider;
        private readonly MLP _mlp;
        private readonly IRandomizer _randomizer;
        private readonly float _p;

        private readonly int _arraySize;

        private float[] _array;
        private int _currentIterationNumber;

        private List<MemFloat[]> _maskMem;
        private int _currentMaskIndex;
        private MemFloat[] _preparedMem;
        public MemFloat[] MaskMem
        {
            get;
            private set;
        }

        private Thread _workThread;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="mlp">Trained MLP</param>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public BigArrayWeightMaskContainer(
            CLProvider clProvider,
            MLP mlp,
            IRandomizer randomizer,
            float p = 0.5f)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _clProvider = clProvider;
            _mlp = mlp;

            _randomizer = randomizer;
            _p = p;

            _arraySize = 0;
            for (var li = 1; li < mlp.Layers.Length; li++)
            {
                var w = mlp.Layers[li - 1].Neurons.Length*mlp.Layers[li].Neurons.Length;
                _arraySize += w;
            }
            _arraySize += TotalIterationCount;

            this.CreateInfrastructure();
        }

        private void CreateInfrastructure()
        {
            var layerCount = _mlp.Layers.Length;

            _array = new float[_arraySize];
            _currentIterationNumber = 0;

            InternalRegenerateArray();

            _maskMem = new List<MemFloat[]>();
            for (var mc = 0; mc < 2; mc++)
            {
                var masks = new MemFloat[layerCount];

                for (var cc = 1; cc < layerCount; cc++)
                {
                    masks[cc] = _clProvider.CreateFloatMem(
                        _mlp.Layers[cc].NonBiasNeuronCount * _mlp.Layers[cc].Neurons[0].Weights.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                        MemFlags.CopyHostPtr | MemFlags.ReadOnly);
                }

                _maskMem.Add(masks);
            }

            _currentMaskIndex = 0;

            FillMem();
        }

        public void RegenerateMask()
        {
            if (_workThread != null)
            {
                _workThread.Join();
            }

            this.MaskMem = _preparedMem;

            WriteWorkingMem();

            ThreadFillMem();
        }

        private void ThreadFillMem()
        {
            _workThread = new Thread(FillMem);
            _workThread.Start();
        }

        private void FillMem()
        {
            //заполняемый мем
            var preparingMem = this._maskMem[_currentMaskIndex];

            //заполняем мем по текущему рабочему индексу
            var layerCount = _mlp.Layers.Length;

            //заполняем слои
            //копируем в мемы
            for (var li = 1; li < layerCount; li++)
            {
                var mem = preparingMem[li];

                var arrayIndex = _randomizer.Next(_array.Length - mem.Array.Length);

                Array.Copy(_array, arrayIndex, mem.Array, 0, mem.Array.Length);
            }

            //копируем в подготовленный
            _preparedMem = preparingMem;

            //обновляем рабочие индексы
            _currentIterationNumber++;

            if (_currentIterationNumber >= TotalIterationCount)
            {
                InternalRegenerateArray();

                _currentIterationNumber = 0;
            }

            _currentMaskIndex = 1 - _currentMaskIndex;

        }

        private void WriteWorkingMem()
        {
            var layerCount = _mlp.Layers.Length;

            for (var li = 1; li < layerCount; li++)
            {
                this.MaskMem[li].Write(BlockModeEnum.NonBlocking);
            }
        }

        private void InternalRegenerateArray()
        {
            var brnd = new Bernoulli(_p)
            {
                RandomSource = new Random(_randomizer.Next(1000000))
            };

            for (var i = 0; i < _arraySize; i++)
            {
                _array[i] = brnd.Sample();
            }
        }

    }
}