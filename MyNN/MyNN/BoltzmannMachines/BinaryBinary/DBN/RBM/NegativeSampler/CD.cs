using System;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler
{
    public class CD : IRBMNegativeSampler
    {
        private readonly IRestrictedBoltzmannMachine _rbm;

        public string Name
        {
            get
            {
                return
                    "Contrastive divergence";
            }
        }

        public CD(IRestrictedBoltzmannMachine rbm)
        {
            #region validate

            if (rbm == null)
            {
                throw new ArgumentNullException("rbm");
            }

            #endregion

            _rbm = rbm;
        }

        public void PrepareTrain(
            int batchSize)
        {
            //empty for CD
        }

        public void PrepareBatch()
        {
            //empty for CD
        }

        public void GetNegativeSample(
            int batchIndex,
            int maxGibbsChainLength)
        {
            var randomIndex = this._rbm.Random.Next(this._rbm.RandomCount);

            //vhv
            for (var cdi = 0; cdi < maxGibbsChainLength; cdi++)
            {
                var ifFirst = cdi == 0;
                var ifLast = cdi == (maxGibbsChainLength - 1);

                this._rbm.ComputeVisible
                    .SetKernelArgMem(0, ifFirst ? this._rbm.Hidden0 : this._rbm.Hidden1)
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
        }

        public void BatchFinished()
        {
            //empty for CD
        }


    }
}
