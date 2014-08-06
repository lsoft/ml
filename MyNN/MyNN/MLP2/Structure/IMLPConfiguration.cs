using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Structure
{
    public interface IMLPConfiguration
    {
        ILayerConfiguration[] Layers
        {
            get;
        }

    }

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
