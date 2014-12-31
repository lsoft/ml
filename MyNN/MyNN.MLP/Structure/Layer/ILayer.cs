﻿using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public interface ILayer
    {
        int TotalNeuronCount
        {
            get;
        }

        INeuron[] Neurons
        {
            get;
        }

        IFunction LayerActivationFunction
        {
            get;
        }

        string GetLayerInformation();

        ILayerConfiguration GetConfiguration();
    }
}