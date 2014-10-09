using System;
using System.Threading;
using MathNet.Numerics.Distributions;
using MyNN.Common.Randomizer;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.DropConnect.WeightMask
{
    /// <summary>
    /// Implementation of weight bit mask container for dropconnect backpropagation algorithm.
    /// This container uses mask array with size that greather than weights count.
    /// </summary>
    public class BigArrayWeightMaskContainer : IOpenCLWeightMaskContainer
    {
        /// <summary>
        /// Iteration count between random array recalculcation
        /// </summary>
        private const int TotalIterationCount = 1000;

        /// <summary>
        /// Tail of random array (it improves randomness)
        /// </summary>
        private const int AdditionalPartSize = 10 * 1024 * 1024;

        private readonly IRandomizer _randomizer;
        private readonly float _p;

        private readonly int _arraySize;
        private readonly uint[] _array;

        private readonly uint[] _bitmask;

        private int _currentIterationNumber;

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

        public MemUint MaskMem
        {
            get;
            private set;
        }

        private Thread _refreshThread;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="previousLayerConfiguration">Previous layer configuration</param>
        /// <param name="currentLayerConfiguration">Current layer configuration</param>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public BigArrayWeightMaskContainer(
            CLProvider clProvider,
            ILayerConfiguration previousLayerConfiguration,
            ILayerConfiguration currentLayerConfiguration,
            IRandomizer randomizer,
            float p = 0.5f)
        {
            if (previousLayerConfiguration == null)
            {
                throw new ArgumentNullException("previousLayerConfiguration");
            }
            if (currentLayerConfiguration == null)
            {
                throw new ArgumentNullException("currentLayerConfiguration");
            }
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            this._randomizer = randomizer;
            this._p = p;

            this._bitIndex = 31;

            this._arraySize = previousLayerConfiguration.Neurons.Length * currentLayerConfiguration.Neurons.Length;
            this._arraySize += AdditionalPartSize;

            this._array = new uint[_arraySize];
            this._currentIterationNumber = 0;

            this._bitmask = new uint[32];
            for (var i = 0; i < 32; i++)
            {
                this._bitmask[i] = (uint)(Math.Pow(2, i));
            }

            MaskMem = clProvider.CreateUintMem(
                currentLayerConfiguration.NonBiasNeuronCount * previousLayerConfiguration.Neurons.Length, //without bias neuron at current layer, but include bias neuron at previous layer
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            InternalRegenerateArray();

            FillMem();
        }

        public void RegenerateMask()
        {
            //если битовая маска "не износилась", то просто меняем ее
            //без необходимости мучаться обновлением буфера
            if (++this._bitIndex < 32)
            {
                return;
            }

            //дожидаемся пока актуализация содержимого мема
            //и буфера не закончится
            if (_refreshThread != null)
            {
                _refreshThread.Join();
            }

            //битовая маска "износилась", освежаем всё
            this._bitIndex = 0;

            //записываем мем в память OpenCL устройства
            //(мем был обновлен на предыдущей итерации RegenerateMask)
            WriteRefreshedMem();

            //снова запускаем рефреш внутреннего буфера
            //рефреш внутреннего буфера заключается в том, что
            //1) со случайного места в буфере копируется кусок в мем
            //(это относительно быстрая операция копирования блока памяти)
            //2) (если уже давно не обновляли содержмое буфера), то
            //буфер перегенерируется целиком, чтобы на следующей
            //итерации FillMem замениться на шаге 1
            AsyncFillMem();
        }

        /// <summary>
        /// Асинхронное перезаполнение содержимого мемам из внутреннего буфера
        /// </summary>
        private void AsyncFillMem()
        {
            _refreshThread = new Thread(FillMem);
            _refreshThread.Start();
        }

        /// <summary>
        /// Перезаполнение содержимого мема из внутреннего буфера
        /// </summary>
        private void FillMem()
        {
            var arrayIndex = _randomizer.Next(_array.Length - MaskMem.Array.Length);
            Array.Copy(_array, arrayIndex, MaskMem.Array, 0, MaskMem.Array.Length);

            //обновляем рабочие индексы
            _currentIterationNumber++;

            if (_currentIterationNumber >= TotalIterationCount)
            {
                InternalRegenerateArray();

                _currentIterationNumber = 0;
            }
        }

        /// <summary>
        /// Запись мема в память OpenCL устройства
        /// </summary>
        private void WriteRefreshedMem()
        {
            this.MaskMem.Write(BlockModeEnum.NonBlocking);
        }

        /// <summary>
        /// Освежение внутреннего буфера с помощью распределения Бернулли
        /// </summary>
        private void InternalRegenerateArray()
        {
            var brnd = new Bernoulli(_p)
            {
                RandomSource = new Random(_randomizer.Next(1000000000))
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
