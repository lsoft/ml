using System;
using MyNN.MLP2.Structure.Layer.Factory.WeightLoader;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Layer.Factory
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
            int withoutBiasNeuronCount)
        {
            return 
                new Layer(
                    _neuronFactory,
                    withoutBiasNeuronCount);
        }

        public ILayer CreateLayer(
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount,
            bool isNeedBiasNeuron,
            bool isPreviousLayerHadBiasNeuron
            )
        {
            var result = new Layer(
                _neuronFactory,
                activationFunction,
                currentLayerNeuronCount,
                previousLayerNeuronCount,
                isNeedBiasNeuron,
                isPreviousLayerHadBiasNeuron);

            return result;
        }


    }
}