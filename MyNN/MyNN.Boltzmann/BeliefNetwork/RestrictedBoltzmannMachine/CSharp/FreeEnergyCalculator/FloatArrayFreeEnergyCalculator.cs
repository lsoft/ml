using System;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator;
using MyNN.Common.Data;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator
{
    public class FloatArrayFreeEnergyCalculator : IFreeEnergyCalculator
    {
        private readonly int _visibleNeuronCount;
        private readonly int _hiddenNeuronCount;
        private readonly int _visibleNeuronCountWithBias;
        private readonly int _hiddenNeuronCountWithBias;

        public FloatArrayFreeEnergyCalculator(
            int visibleNeuronCount,
            int hiddenNeuronCount
            )
        {
            _visibleNeuronCount = visibleNeuronCount;
            _hiddenNeuronCount = hiddenNeuronCount;
            _visibleNeuronCountWithBias = visibleNeuronCount + 1; //bias neuron
            _hiddenNeuronCountWithBias = hiddenNeuronCount + 1; //bias neuron

        }

        public double CalculateFreeEnergy(
            float[] weights,
            IDataSet data)
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            //algorithm has taken from a book "A Practical Guide to Training Restricted Boltzmann Machines"
            //section 16.1

            var freeEnergyArray = new double[data.Count];

            Parallel.For(0, data.Count, vii =>
            //for(var vii = 0; vii < data.Count; vii++)
            {
                var vd = data[vii];
                var freeEnergy = CalculateForDataItem(weights, vd);

                freeEnergyArray[vii] = freeEnergy;
            }
            ); //Parallel.For

            var sumFreeEnergy = freeEnergyArray.Sum();

            return sumFreeEnergy;
        }

        private float CalculateForDataItem(
            float[] weights, 
            IDataItem vd)
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (vd == null)
            {
                throw new ArgumentNullException("vd");
            }

            var vis = 0f;
            for (var i = 0; i < _visibleNeuronCount; i++)
            {
                var vi = vd.Input[i];
                var ai = weights[CalculateWeightIndex(_hiddenNeuronCount, i)];

                var visPart = vi*ai;
                vis += visPart;
            }

            var hid = 0f;
            for (var j = 0; j < _hiddenNeuronCount; j++)
            {
                var xj = CalculateXj(weights, vd, j);

                var expxj = Math.Exp(xj);
                var hidPart = Math.Log(1 + expxj);

                hid += (float) hidPart;
            }

            var freeEnergy = -vis - hid;
            return freeEnergy;
        }

        private float CalculateXj(
            float[] weights, 
            IDataItem vd, 
            int j)
        {
            var xj = 0f;
            for (var i = 0; i < _visibleNeuronCountWithBias; i++)
            {
                var vi =
                    i < _visibleNeuronCount
                        ? vd.Input[i]
                        : 1f;
                var wij = weights[CalculateWeightIndex(j, i)];

                xj += vi*wij;
            }
            return xj;
        }

        private int CalculateWeightIndex(
            int hiddenIndex,
            int visibleIndex
            )
        {
            return
                hiddenIndex * _visibleNeuronCountWithBias + visibleIndex;
        }
    }
}