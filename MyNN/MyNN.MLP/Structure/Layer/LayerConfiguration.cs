using System;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public class LayerConfiguration : ILayerConfiguration
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
            IFunction layerActivationFunction,
            IDimension spatialDimension,
            int weightCount,
            int biasCount,
            INeuronConfiguration[] neurons
            )
        {
            //layerActivationFunction allowed to be null (for input layer)
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }
            if (neurons == null)
            {
                throw new ArgumentNullException("neurons");
            }

            LayerActivationFunction = layerActivationFunction;
            SpatialDimension = spatialDimension;
            WeightCount = weightCount;
            BiasCount = biasCount;
            Neurons = neurons;
        }
    }
}