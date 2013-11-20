using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.NeuralNet.Structure.Neurons.Function;

namespace MyNN.NeuralNet.Structure.Layers
{
    public class LayerInfo
    {
        public int LayerSize
        {
            get;
            private set;
        }

        public IFunction ActivationFunction
        {
            get;
            private set;
        }


        public LayerInfo(int layerSize, IFunction activationFunction)
        {
            if (layerSize <= 0)
            {
                throw new ArgumentException("layerSize");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            LayerSize = layerSize;
            ActivationFunction = activationFunction;
        }
    }
}
