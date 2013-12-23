using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler.ParallelTempering;
using MyNN.Data;

using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler
{
    public class PT : IRBMNegativeSampler
    {
        /// <summary>
        /// Правило применения температуры к цепи
        /// </summary>
        public enum TemperatureApplyRuleEnum
        {
            /// <summary>
            /// применять температуру ко всем весам
            /// </summary>
            All,

            /// <summary>
            /// применять температуру только к весам видимые-скрытые, без весов на bias нейроны
            /// </summary>
            WithoutBias
        }

        private readonly IRestrictedBoltzmannMachine _rbm;

        private readonly List<float> _temperatureList; 
        private int _temperatureCount
        {
            get
            {
                return
                    _temperatureList.Count;
            }
        }
        private readonly TemperatureApplyRuleEnum _temperatureApplyRule;

        private Mem<float> _visibleForEnergy;
        private Mem<float> _summatorForEnergy;

        private List<Mem<float>> _ptChainList;
        private List<Mem<float>> _ptWeightList;

        private readonly Kernel _rescaleWeights, _copyAndScale, _sampleValues;

        private int _total,
                    _swaps;

        public string Name
        {
            get
            {
                return
                    "Parallel tempering";
            }
        }

        public PT(
            IRestrictedBoltzmannMachine rbm, 
            TemperatureApplyRuleEnum temperatureApplyRule, 
            ITemperature temperature)
        {
            #region validate

            if (rbm == null)
            {
                throw new ArgumentNullException("rbm");
            }

            if (temperature == null)
            {
                throw new ArgumentNullException("temperature");
            }

            #endregion

            _rbm = rbm;
            _temperatureApplyRule = temperatureApplyRule;

            //создаем кернелы
            var kernelsSource =
                this._kernelsSource.Replace("{0}", _rbm.VisibleNeuronCount.ToString());

            _rescaleWeights = _rbm.CLProvider.CreateKernel(kernelsSource, "RescaleWeights");
            _sampleValues = _rbm.CLProvider.CreateKernel(kernelsSource, "SampleValues");
            _copyAndScale = _rbm.CLProvider.CreateKernel(kernelsSource, "CopyAndScale");

            _temperatureList = temperature.GetTemperatureList();
        }

        public void PrepareTrain(
            int batchSize)
        {
            #region validate

            if (batchSize <= 0)
            {
                throw new ArgumentException("batchSize <= 0");
            }

            #endregion

            _visibleForEnergy = _rbm.CLProvider.CreateFloatMem(_rbm.VisibleNeuronCount, Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
            _visibleForEnergy.Write(BlockModeEnum.NonBlocking);

            _summatorForEnergy = _rbm.CLProvider.CreateFloatMem(_rbm.VisibleNeuronCount, Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
            _summatorForEnergy.Write(BlockModeEnum.NonBlocking);

            #region готовим pt gibbs chains

            _ptChainList = new List<Mem<float>>();
            for (var c = 0; c < _temperatureCount; c++)
            {
                var chain = _rbm.CLProvider.CreateFloatMem(_rbm.HiddenNeuronCount, Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);

                //очищаем состояние
                Array.Clear(chain.Array, 0, _rbm.HiddenNeuronCount);

                //заполняем случайными числами
                for (var cc = 0; cc < _rbm.HiddenNeuronCount - 1; cc++)
                {
                    chain.Array[cc] = _rbm.Random.NextDouble() < 0.5 ? 0f : 1f;
                }

                //заполняем bias выходных объектов
                chain.Array[_rbm.HiddenNeuronCount - 1] = 1f;

                chain.Write(BlockModeEnum.NonBlocking);

                _ptChainList.Add(chain);
            }

            #endregion

            #region Готовим мемы для весов

            _ptWeightList = new List<Mem<float>>();
            for (var ti = 0; ti < _temperatureCount; ti++)
            {
                var weight = _rbm.CLProvider.CreateFloatMem(_rbm.HiddenNeuronCount * _rbm.VisibleNeuronCount, Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);

                //очищаем состояние
                Array.Clear(weight.Array, 0, weight.Array.Length);

                weight.Write(BlockModeEnum.NonBlocking);

                _ptWeightList.Add(weight);
            }

            #endregion
        }

        public void PrepareBatch()
        {
            //генерируем веса для эпохи
            var ti = 1;
            foreach(var temperature in _temperatureList)
            {
                if (this._temperatureApplyRule == TemperatureApplyRuleEnum.All)
                {
                    this._copyAndScale
                        .SetKernelArgMem(0, this._rbm.Weights)
                        .SetKernelArgMem(1, this._ptWeightList[ti - 1])

                        .SetKernelArg(2, 4, temperature)
                        .SetKernelArg(3, 4, this._rbm.VisibleNeuronCount*this._rbm.HiddenNeuronCount)

                        .EnqueueNDRangeKernel(1);
                }
                else if (this._temperatureApplyRule == TemperatureApplyRuleEnum.WithoutBias)
                {
                    this._rescaleWeights
                        .SetKernelArgMem(0, this._rbm.Weights)
                        .SetKernelArgMem(1, this._ptWeightList[ti - 1])

                        .SetKernelArg(2, 4, temperature)
                        .SetKernelArg(3, 4, this._rbm.VisibleNeuronCount)
                        .SetKernelArg(4, 4, this._rbm.HiddenNeuronCount)

                        .EnqueueNDRangeKernel(1); //!!! векторизовать этот кернел
                }

                ti++;
            }

            //ждем завершения
            this._rbm.CLProvider.QueueFinish();
        }

        public void GetNegativeSample(
            int batchIndex,
            int maxGibbsChainLength)
        {
            //выполняем просчет цепей для всех температур
            for (var ti = 0; ti < this._temperatureCount; ti++)
            {
                var data = this._ptChainList[ti];
                var weights = this._ptWeightList[ti];

                #region осуществляем семплирование по гиббсу для этой цепи

                //vhv
                for (var cdi = 0; cdi < maxGibbsChainLength; cdi++)
                {
                    var randomIndex = this._rbm.Random.Next(this._rbm.RandomCount);

                    var ifFirst = cdi == 0;
                    var ifLast = cdi == (maxGibbsChainLength - 1);

                    this._rbm.ComputeVisible
                        .SetKernelArgMem(0, ifFirst ? data : this._rbm.Hidden1)
                        .SetKernelArgMem(1, this._rbm.Visible)

                        .SetKernelArgMem(2, weights)

                        .SetKernelArg(3, 4, this._rbm.HiddenNeuronCount)
                        .SetKernelArg(4, 4, this._rbm.VisibleNeuronCount)

                        .EnqueueNDRangeKernel(this._rbm.VisibleNeuronCount - 1); //without bias

                    if (ifLast)
                    {
                        this._rbm.ComputeHidden
                            .SetKernelArgMem(0, this._rbm.Hidden1)
                            .SetKernelArgMem(1, this._rbm.Visible)

                            .SetKernelArgMem(2, weights)

                            .SetKernelArg(3, 4, this._rbm.HiddenNeuronCount)
                            .SetKernelArg(4, 4, this._rbm.VisibleNeuronCount)

                            .EnqueueNDRangeKernel(this._rbm.HiddenNeuronCount - 1); //without bias
                    }
                    else
                    {
                        this._rbm.SampleHidden
                            .SetKernelArgMem(0, this._rbm.Hidden1)
                            .SetKernelArgMem(1, this._rbm.Visible)

                            .SetKernelArgMem(2, weights)
                            .SetKernelArgMem(3, this._rbm.Randoms)

                            .SetKernelArg(4, 4, this._rbm.HiddenNeuronCount)
                            .SetKernelArg(5, 4, this._rbm.VisibleNeuronCount)

                            .SetKernelArg(6, 4, randomIndex)
                            .SetKernelArg(7, 4, this._rbm.RandomCount)

                            .EnqueueNDRangeKernel(this._rbm.HiddenNeuronCount - 1); //without bias
                    }
                }

                //семплируем вероятности в состояния

                #region validate

                if (this._rbm.RandomCount <= this._rbm.HiddenNeuronCount - 1)
                {
                    throw new InvalidOperationException("Мало рандомов сгенерировано, сделайте количество рандомов больше чем (число скрытых нейронов + 1)");
                }

                #endregion

                var randomIndex2 = this._rbm.Random.Next(this._rbm.RandomCount - this._rbm.HiddenNeuronCount - 1);

                _sampleValues
                    .SetKernelArgMem(0, this._rbm.Hidden1)
                    .SetKernelArgMem(1, data)

                    .SetKernelArgMem(2, this._rbm.Randoms)

                    .SetKernelArg(3, 4, randomIndex2)

                    .EnqueueNDRangeKernel(this._rbm.HiddenNeuronCount - 1); //without bias

                ////копируем без семплирования (с копированием без семплирования работает тоже, но лучше с семплированием (надежнее))
                //this._copyAndScale
                //    .SetKernelArgMem(0, this._rbm.Hidden1)
                //    .SetKernelArgMem(1, data)

                //    .SetKernelArg(2, 4, 1f)
                //    .SetKernelArg(3, 4, this._rbm.HiddenNeuronCount - 1)

                //    .EnqueueNDRangeKernel(1);

                #endregion

                //семплирование осуществили, теперь в data значение при данной температуре
            }

            //в hidden1 будет значение нужной нам цепи, так как
            //цикл просчета цепей работает от горячей к нужной нам (нужная нам - последняя)

            //возращается значение цепи без замены итема
            //замененный итем сработает на следующей итерации
        }

        public void BatchFinished()
        {
            //меняем цепи

            #region осуществляем ротацию значений в цепях с разной температурой

            var temperatureList = Enumerable
                .Range(1, _temperatureCount)
                .Select(j => j / (float)(_temperatureCount))
                .ToList();

            for (var leftIndex = 0; leftIndex < this._temperatureCount - 1; leftIndex++)
            {
                var leftData = this._ptChainList[leftIndex];
                var leftWeights = this._ptWeightList[leftIndex];
                var leftTemperature = temperatureList[leftIndex];

                var rightData = this._ptChainList[leftIndex + 1];
                var rightWeights = this._ptWeightList[leftIndex + 1];
                var rightTemperature = temperatureList[leftIndex + 1];

                var randomIndex0 = this._rbm.Random.Next(this._rbm.RandomCount);

                //семплируем в визибл от полученного значения в температурной цепи
                _rbm.SampleVisible
                    .SetKernelArgMem(0, leftData)
                    .SetKernelArgMem(1, this._visibleForEnergy)

                    .SetKernelArgMem(2, leftWeights)
                    .SetKernelArgMem(3, _rbm.Randoms)

                    .SetKernelArg(4, 4, _rbm.HiddenNeuronCount)
                    .SetKernelArg(5, 4, _rbm.VisibleNeuronCount)

                    .SetKernelArg(6, 4, randomIndex0)
                    .SetKernelArg(7, 4, _rbm.RandomCount)

                    .EnqueueNDRangeKernel(_rbm.VisibleNeuronCount - 1); //without bias

                //this._rbm.ComputeVisible
                //    .SetKernelArgMem(0, leftData)
                //    .SetKernelArgMem(1, this._visibleForEnergy)

                //    .SetKernelArgMem(2, leftWeights)

                //    .SetKernelArg(3, 4, this._rbm.HiddenNeuronCount)
                //    .SetKernelArg(4, 4, this._rbm.VisibleNeuronCount)

                //    .EnqueueNDRangeKernel(this._rbm.VisibleNeuronCount - 1); //without bias

                _visibleForEnergy.Read(BlockModeEnum.Blocking);

                throw new NotImplementedException("Тут надо прибираться, так как константа 784 и 10 прям в коде");

                var p0x0 = _rbm.CalculateFreeEnergySet(
                    leftWeights,
                    new DataSet(
                        new List<DataItem>
                        {
                            new DataItem(
                                _visibleForEnergy.Array.Take(784).ToArray(),
                                new float[10]),
                        },
                        null));

                var p1x0 = _rbm.CalculateFreeEnergySet(
                    rightWeights,
                    new DataSet(
                        new List<DataItem>
                        {
                            new DataItem(
                                _visibleForEnergy.Array.Take(784).ToArray(),
                                new float[10]),
                        },
                        null));

                var randomIndex1 = this._rbm.Random.Next(this._rbm.RandomCount);

                //семплируем в визибл от полученного значения в температурной цепи
                //_rbm.SampleVisible
                //    .SetKernelArgMem(0, rightData)
                //    .SetKernelArgMem(1, this._visibleForEnergy)

                //    .SetKernelArgMem(2, rightWeights)
                //    .SetKernelArgMem(3, _rbm.Randoms)

                //    .SetKernelArg(4, 4, _rbm.HiddenNeuronCount)
                //    .SetKernelArg(5, 4, _rbm.VisibleNeuronCount)

                //    .SetKernelArg(6, 4, randomIndex1)
                //    .SetKernelArg(7, 4, _rbm.RandomCount)

                //    .EnqueueNDRangeKernel(_rbm.VisibleNeuronCount - 1); //without bias

                this._rbm.ComputeVisible
                    .SetKernelArgMem(0, rightData)
                    .SetKernelArgMem(1, this._visibleForEnergy)

                    .SetKernelArgMem(2, rightWeights)

                    .SetKernelArg(3, 4, this._rbm.HiddenNeuronCount)
                    .SetKernelArg(4, 4, this._rbm.VisibleNeuronCount)

                    .EnqueueNDRangeKernel(this._rbm.VisibleNeuronCount - 1); //without bias

                _visibleForEnergy.Read(BlockModeEnum.Blocking);

                throw new NotImplementedException("Тут надо прибираться, так как константа 784 и 10 прям в коде");

                var p0x1 = _rbm.CalculateFreeEnergySet(
                    leftWeights,
                    new DataSet(
                        new List<DataItem>
                        {
                            new DataItem(
                                _visibleForEnergy.Array.Take(784).ToArray(),
                                new float[10])
                        },
                        null));

                var p1x1 = _rbm.CalculateFreeEnergySet(
                    rightWeights,
                    new DataSet(
                        new List<DataItem>
                        {
                            new DataItem(
                                _visibleForEnergy.Array.Take(784).ToArray(),
                                new float[10])
                        },
                        null));

                //var leftEnergy = this.CalculateEnergy(leftData, leftWeights);
                //var rightEnergy = this.CalculateEnergy(rightData, rightWeights);

                //var ur = (leftTemperature - rightTemperature) * (p0x0 - p1x1);
                var ur = - p0x1 - p1x0 + p0x0 + p1x1;

                var r = Math.Min(
                    1,
                    Math.Exp(ur));

                if (_rbm.Random.NextDouble() < r)
                {
                    //меняем местами итемы
                    var temp0 = _ptChainList[leftIndex];
                    _ptChainList[leftIndex] = _ptChainList[leftIndex + 1];
                    _ptChainList[leftIndex + 1] = temp0;

                    _swaps++;
                }

                _total++;

                if (_total == 100)
                {
                    Console.WriteLine(
                        "Swaps percent {0}%",
                        _swaps);

                    _total = 0;
                    _swaps = 0;
                }
            }

            #endregion

        }

        private readonly string _kernelsSource = @"
typedef struct
{
    float VWeight[{0}];
} VisibleWeight;

__kernel void RescaleWeights(
    __global float * sourceWeight,
    __global float * destWeight,
    float temperature,
    int visibleNeuronCount, //with bias
    int hiddenNeuronCount) //with bias
        //!!! векторизовать этот кернел
{
    //int kernelIndex = get_global_id(0);

    for(int cc = 0; cc< visibleNeuronCount * hiddenNeuronCount; cc++)
    {
        destWeight[cc] = 
            sourceWeight[cc] * temperature; //!!! проверить что быстрее ---> mad(sourceWeight[cc], temperature, 0.0);
    }

    for(int h = 0; h < hiddenNeuronCount - 1; h++)
    {
        int index = (h + 1) * visibleNeuronCount - 1;

        destWeight[index] = sourceWeight[index];
    }

    for(int v = 0; v < visibleNeuronCount; v++)
    {
        int index = (hiddenNeuronCount - 1) * visibleNeuronCount + v;

        destWeight[index] = sourceWeight[index];
    }
}

__kernel void CopyAndScale(
    __global float * source,
    __global float * dest,
    float temperature,
    int length) //!!! проверить что после векторизации ничего не сломалось (что все элементы скалятся нормально)
{
    for(int cc4 = 0; cc4 < length / 4; cc4++)
    {
        vstore4(
            vload4(cc4, source) * temperature,
            cc4,
            dest);
    }

    for(int cc = length - length % 4; cc < length; cc++)
    {
        dest[cc] = 
            source[cc] * temperature;
    }
}

__kernel void SampleValues(
    __global float * hidden1,
    __global float * pcd,
    __global float * randoms,

    int randomIndex)
{
    int kernelIndex = get_global_id(0);

    //уникальный рандом для каждого work unit
    float random = randoms[kernelIndex + randomIndex];

    pcd[kernelIndex] = random <= hidden1[kernelIndex] ? 1 : 0;
}

";

    }
}
