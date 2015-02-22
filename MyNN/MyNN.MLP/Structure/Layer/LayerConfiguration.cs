using System;
using MyNN.MLP.Structure.Neuron;

namespace MyNN.MLP.Structure.Layer
{
    public class LayerConfiguration : ILayerConfiguration
    {
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
            get
            {
                return
                    SpatialDimension.Multiplied;
            }
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

        public LayerConfiguration(
            IDimension spatialDimension,
            int weightCount,
            int biasCount,
            INeuronConfiguration[] neurons
            )
        {
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }
            if (neurons == null)
            {
                throw new ArgumentNullException("neurons");
            }

            SpatialDimension = spatialDimension;
            WeightCount = weightCount;
            BiasCount = biasCount;
            Neurons = neurons;
        }
    }
}