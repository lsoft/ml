using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.Tests
{
    internal class TestLayerConfiguration : ILayerConfiguration
    {
        public IFunction LayerActivationFunction
        {
            get;
            private set;
        }

        public IDimension SpatialDimension
        {
            get;
            private set;
        }

        public INeuronConfiguration[] Neurons
        {
            get;
            private set;
        }

        public int TotalNeuronCount
        {
            get;
            private set;
        }

        public int WeightCount
        {
            get;
            private set;
        }

        public int BiasCount
        {
            get;
            private set;
        }

        public TestLayerConfiguration(
            int totalNeuronCount,
            int weightCount,
            int biasCount
            )
        {
            this.LayerActivationFunction = null;
            this.SpatialDimension = new Dimension(1, totalNeuronCount);
            this.Neurons = new INeuronConfiguration[totalNeuronCount];

            this.TotalNeuronCount = totalNeuronCount;
            this.WeightCount = weightCount;
            this.BiasCount = biasCount;
        }
    }
}
