using System;
using MyNN.MLP.Structure.Neuron;

namespace MyNN.MLP.Structure.Layer
{
    public class LayerConfiguration : ILayerConfiguration
    {
        public INeuronConfiguration[] Neurons
        {
            get;
            private set;
        }

        public int TotalNeuronCount
        {
            get;
            private set;
        }

        public LayerConfiguration(
            INeuronConfiguration[] neurons, 
            int totalNeuronCount
            )
        {
            if (neurons == null)
            {
                throw new ArgumentNullException("neurons");
            }

            Neurons = neurons;
            TotalNeuronCount = totalNeuronCount;
        }
    }
}