using System;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public class AvgPoolingLayerConfiguration : IAvgPoolingLayerConfiguration
    {
        public IFunction LayerActivationFunction
        {
            get
            {
                throw new NotSupportedException("Не поддерживается для пулинг-слоя");
            }
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

        public int FeatureMapCount
        {
            get;
            private set;
        }

        public int InverseScaleFactor
        {
            get
            {
                var fisf = 1f / this.ScaleFactor;

                if (Math.Abs(fisf - (int)fisf) > float.Epsilon)
                {
                    throw new ArgumentException("Invalid scale factor");
                }

                var isf = (int)fisf;

                return isf;
            }
        }

        public float ScaleFactor
        {
            get;
            private set;
        }

        public AvgPoolingLayerConfiguration(
            IDimension spatialDimension,
            int featureMapCount,
            float scaleFactor,
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
            FeatureMapCount = featureMapCount;
            ScaleFactor = scaleFactor;
            WeightCount = weightCount;
            BiasCount = biasCount;
            Neurons = neurons;
        }

    }
}