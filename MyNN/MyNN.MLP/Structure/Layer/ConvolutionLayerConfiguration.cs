using System;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public class ConvolutionLayerConfiguration : IConvolutionLayerConfiguration
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
                    FeatureMapCount * SpatialDimension.Multiplied;
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

        public IDimension KernelSpatialDimension
        {
            get;
            private set;
        }

        public int FeatureMapCount
        {
            get;
            private set;
        }

        public ConvolutionLayerConfiguration(
            IFunction layerActivationFunction,
            IDimension spatialDimension,
            int featureMapCount,
            IDimension kernelSpatialDimension,
            int weightCount,
            int biasCount,
            INeuronConfiguration[] neurons
            )
        {
            if (layerActivationFunction == null)
            {
                throw new ArgumentNullException("layerActivationFunction");
            }
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }
            if (kernelSpatialDimension == null)
            {
                throw new ArgumentNullException("kernelSpatialDimension");
            }
            if (neurons == null)
            {
                throw new ArgumentNullException("neurons");
            }

            LayerActivationFunction = layerActivationFunction;
            SpatialDimension = spatialDimension;
            FeatureMapCount = featureMapCount;
            KernelSpatialDimension = kernelSpatialDimension;
            WeightCount = weightCount;
            BiasCount = biasCount;
            Neurons = neurons;
        }
    }
}