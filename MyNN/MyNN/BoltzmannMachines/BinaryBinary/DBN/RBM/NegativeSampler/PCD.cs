using System;
using System.Collections.Generic;
using MyNN.OpenCL;
using MyNN.OpenCL.Mem;
using OpenCL.Net;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler
{
    public class PCD : IRBMNegativeSampler
    {
        private readonly IRestrictedBoltzmannMachine _rbm;

        private List<Mem<float>> _pcdChainList;

        private readonly Kernel _updatePersistent;

        public string Name
        {
            get
            {
                return
                    "Persistent contrastive divergence";
            }
        }

        public PCD(IRestrictedBoltzmannMachine rbm)
        {
            if (rbm == null)
            {
                throw new ArgumentNullException("rbm");
            }

            _rbm = rbm;

            //создаем кернелы
            _updatePersistent = _rbm.CLProvider.CreateKernel(_kernelsSource, "UpdatePersistentChain");
        }

        public void PrepareTrain(
            int batchSize)
        {
            #region готовим pcd gibbs chains

            _pcdChainList = new List<Mem<float>>();
            for (var c = 0; c < batchSize; c++)
            {
                var chain = _rbm.CLProvider.CreateFloatMem(_rbm.HiddenNeuronCount, Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);

                //очищаем состояние
                Array.Clear(chain.Array, 0, _rbm.HiddenNeuronCount);

                //заполняем bias выходных объектов
                chain.Array[_rbm.HiddenNeuronCount - 1] = 1f;

                chain.Write(BlockModeEnum.NonBlocking);

                _pcdChainList.Add(chain);
            }

            #endregion
        }

        public void PrepareBatch()
        {
            //empty for PCD
        }
        
        public void GetNegativeSample(
            int batchIndex,
            int maxGibbsChainLength)
        {
            //vhv
            for (var cdi = 0; cdi < maxGibbsChainLength; cdi++)
            {
                var randomIndex = this._rbm.Random.Next(this._rbm.RandomCount);

                var ifFirst = cdi == 0;
                var ifLast = cdi == (maxGibbsChainLength - 1);

                this._rbm.ComputeVisible
                    .SetKernelArgMem(0, ifFirst ? _pcdChainList[batchIndex] : this._rbm.Hidden1)
                    .SetKernelArgMem(1, this._rbm.Visible)

                    .SetKernelArgMem(2, this._rbm.Weights)

                    .SetKernelArg(3, 4, this._rbm.HiddenNeuronCount)
                    .SetKernelArg(4, 4, this._rbm.VisibleNeuronCount)

                    .EnqueueNDRangeKernel(this._rbm.VisibleNeuronCount - 1); //without bias

                if (ifLast)
                {
                    this._rbm.ComputeHidden
                        .SetKernelArgMem(0, this._rbm.Hidden1)
                        .SetKernelArgMem(1, this._rbm.Visible)

                        .SetKernelArgMem(2, this._rbm.Weights)

                        .SetKernelArg(3, 4, this._rbm.HiddenNeuronCount)
                        .SetKernelArg(4, 4, this._rbm.VisibleNeuronCount)

                        .EnqueueNDRangeKernel(this._rbm.HiddenNeuronCount - 1); //without bias
                }
                else
                {
                    this._rbm.SampleHidden
                        .SetKernelArgMem(0, this._rbm.Hidden1)
                        .SetKernelArgMem(1, this._rbm.Visible)

                        .SetKernelArgMem(2, this._rbm.Weights)
                        .SetKernelArgMem(3, this._rbm.Randoms)

                        .SetKernelArg(4, 4, this._rbm.HiddenNeuronCount)
                        .SetKernelArg(5, 4, this._rbm.VisibleNeuronCount)

                        .SetKernelArg(6, 4, randomIndex)
                        .SetKernelArg(7, 4, this._rbm.RandomCount)

                        .EnqueueNDRangeKernel(this._rbm.HiddenNeuronCount - 1); //without bias
                }
            }

            //обновляем персистентную цепь

            #region validate

            if (this._rbm.RandomCount <= this._rbm.HiddenNeuronCount - 1)
            {
                throw new InvalidOperationException("Мало рандомов сгенерировано, сделайте количество рандомов больше чем (число скрытых нейронов + 1)");
            }

            #endregion

            var randomIndex2 = this._rbm.Random.Next(this._rbm.RandomCount - this._rbm.HiddenNeuronCount - 1);

            _updatePersistent
                .SetKernelArgMem(0, this._rbm.Hidden1)
                .SetKernelArgMem(1, _pcdChainList[batchIndex])

                .SetKernelArgMem(2, this._rbm.Randoms)

                .SetKernelArg(3, 4, randomIndex2)

                .EnqueueNDRangeKernel(this._rbm.HiddenNeuronCount - 1); //without bias

        }

        public void BatchFinished()
        {
            //empty for PCD
        }


        private readonly string _kernelsSource = @"
__kernel void UpdatePersistentChain( //!!! на ЦП поддается значительному ускорению, если не обрабатывать по 1 флоату на кернел
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
