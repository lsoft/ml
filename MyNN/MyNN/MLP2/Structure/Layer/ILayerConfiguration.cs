using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.Structure.Neurons;

namespace MyNN.MLP2.Structure.Layer
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
