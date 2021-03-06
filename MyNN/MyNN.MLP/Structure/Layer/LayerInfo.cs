﻿using System;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
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
