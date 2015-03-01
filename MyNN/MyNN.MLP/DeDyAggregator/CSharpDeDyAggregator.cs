using System;
using System.Threading.Tasks;
using MyNN.Common.Other;

namespace MyNN.MLP.DeDyAggregator
{
    public class CSharpDeDyAggregator : ICSharpDeDyAggregator
    {
        private readonly int _previousLayerNeuronCount;
        private readonly int _aggregateLayerNeuronCount;
        private readonly float[] _aggregateLayerWeights;

        public float[] DeDz
        {
            get;
            private set;
        }

        public float[] DeDy
        {
            get;
            private set;
        }

        public int TotalNeuronCount
        {
            get
            {
                return
                    _aggregateLayerNeuronCount;
            }
        }

        public CSharpDeDyAggregator(
            int previousLayerNeuronCount,
            int aggregateLayerNeuronCount,
            float[] aggregateLayerWeights
            )
        {
            if (aggregateLayerWeights == null)
            {
                throw new ArgumentNullException("aggregateLayerWeights");
            }
            if ((previousLayerNeuronCount * aggregateLayerNeuronCount) != aggregateLayerWeights.Length)
            {
                throw new ArgumentException("(previousLayerNeuronCount * aggregateLayerNeuronCount) != aggregateLayerWeights.Length");
            }

            _previousLayerNeuronCount = previousLayerNeuronCount;
            _aggregateLayerNeuronCount = aggregateLayerNeuronCount;
            _aggregateLayerWeights = aggregateLayerWeights;

            this.DeDz = new float[aggregateLayerNeuronCount];
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
                    float dedz = this.DeDz[aggregateNeuronIndex];
                    float dedy = w * dedz;

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
            this.DeDz.Clear();
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
