using System;
using MyNN.MLP.Structure.Neuron;

namespace MyNN.MLP.Structure.Layer
{
    public interface ILayerConfiguration
    {
        INeuronConfiguration[] Neurons
        {
            get;
        }

        bool IsBiasNeuronExists
        {
            get;
        }

        int NonBiasNeuronCount
        {
            get;
        }
    }

    public class LayerConfiguration : ILayerConfiguration
    {
        public INeuronConfiguration[] Neurons
        {
            get;
            private set;
        }

        public bool IsBiasNeuronExists
        {
            get;
            private set;
        }

        public int NonBiasNeuronCount
        {
            get;
            private set;
        }

        public LayerConfiguration(
            INeuronConfiguration[] neurons, 
            bool isBiasNeuronExists, 
            int nonBiasNeuronCount)
        {
            if (neurons == null)
            {
                throw new ArgumentNullException("neurons");
            }

            Neurons = neurons;
            IsBiasNeuronExists = isBiasNeuronExists;
            NonBiasNeuronCount = nonBiasNeuronCount;
        }
    }
}
