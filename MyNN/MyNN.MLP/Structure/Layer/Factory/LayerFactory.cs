using System;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer.Factory
{
    [Serializable]
    public class LayerFactory : ILayerFactory
    {
        private readonly INeuronFactory _neuronFactory;

        public LayerFactory(
            INeuronFactory neuronFactory)
        {
            if (neuronFactory == null)
            {
                throw new ArgumentNullException("neuronFactory");
            }

            _neuronFactory = neuronFactory;
        }

        public ILayer CreateInputLayer(
            int totalNeuronCount
            )
        {
            return
                new Layer(
                    _neuronFactory,
                    totalNeuronCount
                    );
        }

        public ILayer CreateLayer(
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount
            )
        {
            var result = new Layer(
                _neuronFactory,
                activationFunction,
                currentLayerNeuronCount,
                previousLayerNeuronCount
                );

            return result;
        }


    }
}