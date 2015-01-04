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
                    SpatialDimension.TotalNeuronCount;
            }
        }

        public LayerConfiguration(
            IDimension spatialDimension,
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
            Neurons = neurons;
        }
    }
}