using System;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Structure
{
    public class MLPConfiguration : IMLPConfiguration
    {
        public ILayerConfiguration[] Layers
        {
            get;
            private set;
        }

        public MLPConfiguration(
            ILayerConfiguration[] layers)
        {
            if (layers == null)
            {
                throw new ArgumentNullException("layers");
            }

            Layers = layers;
        }
    }
}