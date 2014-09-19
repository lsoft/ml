using System;
using System.Collections.Generic;
using System.Threading;
using MathNet.Numerics.Distributions;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.WeightMask
{
    /// <summary>
    /// Implementation of weight bit mask container for dropconnect backpropagation algorithm.
    /// This container uses mask array with size that greather than weights count.
    /// </summary>
    public class BigArrayWeightBitMaskContainer : IOpenCLWeightBitMaskContainer
    {
        /// <summary>
        /// Iteration count between random array recalculcation
        /// </summary>
        private const int TotalIterationCount = 10000;

        /// <summary>
        /// Tail of random array (it improves randomness)
        /// </summary>
        private const int AdditionalPartSize = 100000;

        private readonly CLProvider _clProvider;
        private readonly IMLPConfiguration _mlpConfiguration;
        private readonly IRandomizer _randomizer;
        private readonly float _p;

        private readonly int _arraySize;

        private readonly uint[] _bitmask;

        private uint[] _array;
        private int _currentIterationNumber;

        private List<MemUint[]> _maskMem;
        private int _currentMaskIndex;
        private MemUint[] _preparedMem;

        private uint _bitIndex;
        public uint BitMask
        {
            get
            {
                var result = (uint) Math.Pow(2, _bitIndex);

                return
                    result;
            }
        }

        public MemUint[] MaskMem
        {
            get;
            private set;
        }

        private Thread _workThread;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="mlpConfiguration">Configuration of MLP</param>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public BigArrayWeightBitMaskContainer(
            CLProvider clProvider,
            IMLPConfiguration mlpConfiguration,
            IRandomizer randomizer,
            float p = 0.5f)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlpConfiguration == null)
            {
                throw new ArgumentNullException("mlpConfiguration");
            }
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            _clProvider = clProvider;
            _mlpConfiguration = mlpConfiguration;

            _randomizer = randomizer;
            _p = p;

            this._bitIndex = 31;

            _arraySize = 0;
            for (var li = 1; li < mlpConfiguration.Layers.Length; li++)
            {
                var w = mlpConfiguration.Layers[li - 1].Neurons.Length*mlpConfiguration.Layers[li].Neurons.Length;
                _arraySize += w;
            }
            _arraySize += AdditionalPartSize;

            _bitmask = new uint[32];
            for (var i = 0; i < 32; i++)
            {
                _bitmask[i] = (uint)(Math.Pow(2, i));
            }

            this.CreateInfrastructure();
        }

        private void CreateInfrastructure()
        {
            var layerCount = _mlpConfiguration.Layers.Length;

            _array = new uint[_arraySize];
            _currentIterationNumber = 0;

            InternalRegenerateArray();

            _maskMem = new List<MemUint[]>();
            for (var mc = 0; mc < 2; mc++)
            {
                var masks = new MemUint[layerCount];

                for (var cc = 1; cc < layerCount; cc++)
                {
                    masks[cc] = _clProvider.CreateUintMem(
                        _mlpConfiguration.Layers[cc].NonBiasNeuronCount * _mlpConfiguration.Layers[cc].Neurons[0].WeightsCount, //without bias neuron at current layer, but include bias neuron at previous layer
                        MemFlags.CopyHostPtr | MemFlags.ReadOnly);
                }

                _maskMem.Add(masks);
            }

            _currentMaskIndex = 0;

            FillMem();
        }

        public void RegenerateMask()
        {
            if (++this._bitIndex < 32)
            {
                return;
            }

            if (_workThread != null)
            {
                _workThread.Join();
            }

            this.MaskMem = _preparedMem;
            this._bitIndex = 0;

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
            var layerCount = _mlpConfiguration.Layers.Length;

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
            var layerCount = _mlpConfiguration.Layers.Length;

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
                uint ci = 0;

                for (var cii = 0; cii < 32; cii++)
                {
                    if (brnd.Sample() > 0)
                    {
                        ci |= _bitmask[cii];
                    }
                }

                _array[i] = ci;
            }
        }

    }
}
