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
            IDimension dimension
            )
        {
            if (dimension == null)
            {
                throw new ArgumentNullException("dimension");
            }

            return
                new FullConnectedLayer(
                    _neuronFactory,
                    dimension
                    );
        }

        public ILayer CreateFullConnectedLayer(
            IFunction activationFunction,
            IDimension dimension,
            int previousLayerNeuronCount
            )
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }
            if (dimension == null)
            {
                throw new ArgumentNullException("dimension");
            }

            var result = new FullConnectedLayer(
                _neuronFactory,
                activationFunction,
                dimension,
                previousLayerNeuronCount
                );

            return result;
        }


    }
}