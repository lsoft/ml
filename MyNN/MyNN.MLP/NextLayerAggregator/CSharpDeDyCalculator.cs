using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.Other;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.NextLayerAggregator
{
    public class CSharpDeDyCalculator : ICSharpDeDyCalculator
    {
        private readonly int _previousLayerNeuronCount;
        private readonly int _aggregateLayerNeuronCount;
        private readonly float[] _aggregateLayerDeDz;
        private readonly float[] _aggregateLayerWeights;

        public float[] DeDy
        {
            get;
            private set;
        }

        public CSharpDeDyCalculator(
            int previousLayerNeuronCount,
            int aggregateLayerNeuronCount,
            float[] aggregateLayerDeDz,
            float[] aggregateLayerWeights
            )
        {
            if (aggregateLayerDeDz == null)
            {
                throw new ArgumentNullException("aggregateLayerDeDz");
            }
            if (aggregateLayerWeights == null)
            {
                throw new ArgumentNullException("aggregateLayerWeights");
            }
            if (aggregateLayerNeuronCount != aggregateLayerDeDz.Length)
            {
                throw new ArgumentException("aggregateLayerNeuronCount != aggregateLayerDeDz.Length");
            }
            if ((previousLayerNeuronCount * aggregateLayerNeuronCount) != aggregateLayerWeights.Length)
            {
                throw new ArgumentException("(previousLayerNeuronCount * aggregateLayerNeuronCount) != aggregateLayerWeights.Length");
            }

            _previousLayerNeuronCount = previousLayerNeuronCount;
            _aggregateLayerNeuronCount = aggregateLayerNeuronCount;
            _aggregateLayerDeDz = aggregateLayerDeDz;
            _aggregateLayerWeights = aggregateLayerWeights;

            this.DeDy = new float[_previousLayerNeuronCount];
        }

        public void Aggregate(
            )
        {
            Parallel.For(0, _previousLayerNeuronCount, previousLayerNeuronIndex =>
            //for(var previousLayerNeuronIndex = 0; previousLayerNeuronIndex < _previousLayerNeuronCount; previousLayerNeuronIndex++)
            {
                //просчет состояния нейронов текущего слоя, по состоянию нейронов последующего (with Kahan Algorithm)
                var accDeDy = new KahanAlgorithm.Accumulator();
                for (var aggregateNeuronIndex = 0; aggregateNeuronIndex < _aggregateLayerNeuronCount; ++aggregateNeuronIndex)
                {
                    int nextWeightIndex = ComputeWeightIndex(
                        _previousLayerNeuronCount,
                        aggregateNeuronIndex)
                    + previousLayerNeuronIndex; //не векторизуется:(

                    float w = _aggregateLayerWeights[nextWeightIndex]; //w is a dz/dy
                    float dedz = _aggregateLayerDeDz[aggregateNeuronIndex];
                    float dedy = w*dedz;

                    KahanAlgorithm.AddElement(ref accDeDy, dedy);
                }

                this.DeDy[previousLayerNeuronIndex] = accDeDy.Sum;
            }
            ); //Parallel.For
        }

        public void ClearAndWrite(
            )
        {
            this.DeDy.Clear();
        }

        private static int ComputeWeightIndex(
            int previousLayerNeuronCount,
            int neuronIndex)
        {
            return
                previousLayerNeuronCount * neuronIndex;
        }

    }
}
