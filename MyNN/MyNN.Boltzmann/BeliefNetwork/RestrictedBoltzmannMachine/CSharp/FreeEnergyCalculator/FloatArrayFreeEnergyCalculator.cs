using System;
using System.Linq;
using System.Threading.Tasks;
using MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator
{
    public class FloatArrayFreeEnergyCalculator : IFreeEnergyCalculator
    {
        private readonly int _visibleNeuronCount;
        private readonly int _hiddenNeuronCount;

        public FloatArrayFreeEnergyCalculator(
            int visibleNeuronCount,
            int hiddenNeuronCount
            )
        {
            _visibleNeuronCount = visibleNeuronCount;
            _hiddenNeuronCount = hiddenNeuronCount;
        }

        public double CalculateFreeEnergy(
            float[] weights,
            float[] visibleBiases,
            float[] hiddenBiases,
            IDataSet data
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (visibleBiases == null)
            {
                throw new ArgumentNullException("visibleBiases");
            }
            if (hiddenBiases == null)
            {
                throw new ArgumentNullException("hiddenBiases");
            }
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            //algorithm has taken from a book "A Practical Guide to Training Restricted Boltzmann Machines"
            //section 16.1

            var freeEnergyArray = new double[data.Count];

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount
            };

            Parallel.ForEach(data, parallelOptions, (vd, pls, vii) =>
            {
                var freeEnergy = CalculateForDataItem(
                    weights,
                    visibleBiases,
                    hiddenBiases,
                    vd
                    );

                freeEnergyArray[vii] = freeEnergy;
            });
            //Parallel.For(0, data.Count, vii =>
            //{
            //    var vd = data.Data[vii];
            //    var freeEnergy = CalculateForDataItem(weights, vd);

            //    freeEnergyArray[vii] = freeEnergy;
            //}
            //); //Parallel.For

            var sumFreeEnergy = freeEnergyArray.Sum();

            return sumFreeEnergy;
        }

        private float CalculateForDataItem(
            float[] weights,
            float[] visibleBiases,
            float[] hiddenBiases,
            IDataItem vd
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (visibleBiases == null)
            {
                throw new ArgumentNullException("visibleBiases");
            }
            if (hiddenBiases == null)
            {
                throw new ArgumentNullException("hiddenBiases");
            }
            if (vd == null)
            {
                throw new ArgumentNullException("vd");
            }

            var vis = 0f;
            for (var i = 0; i < _visibleNeuronCount; i++)
            {
                var vi = vd.Input[i];
                var ai = visibleBiases[i];

                var visPart = vi*ai;
                vis += visPart;
            }

            var hid = 0f;
            for (var j = 0; j < _hiddenNeuronCount; j++)
            {
                var xj = CalculateXj(
                    weights,
                    hiddenBiases,
                    vd,
                    j
                    );

                var expxj = Math.Exp(xj);
                var hidPart = Math.Log(1 + expxj);

                hid += (float) hidPart;
            }

            var freeEnergy = -vis - hid;
            return freeEnergy;
        }

        private float CalculateXj(
            float[] weights, 
            float[] hiddenBiases,
            IDataItem vd, 
            int j
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (hiddenBiases == null)
            {
                throw new ArgumentNullException("hiddenBiases");
            }
            if (vd == null)
            {
                throw new ArgumentNullException("vd");
            }

            var xj = 0f;
            for (var i = 0; i < _visibleNeuronCount; i++)
            {
                var vi = vd.Input[i];
                var wij = weights[CalculateWeightIndex(j, i)];

                xj += vi*wij;
            }

            xj += hiddenBiases[j];

            return xj;
        }

        private int CalculateWeightIndex(
            int hiddenIndex,
            int visibleIndex
            )
        {
            return
                hiddenIndex * _visibleNeuronCount + visibleIndex;
        }
    }
}