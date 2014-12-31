using System;
using System.Linq;
using MyNN.Common.Other;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    [Serializable]
    public class Layer : ILayer
    {
        public int TotalNeuronCount
        {
            get
            {
                return
                    this.Neurons.Length;
            }
        }

        public INeuron[] Neurons
        {
            get;
            private set;
        }

        public IFunction LayerActivationFunction
        {
            get;
            private set;
        }

        /// <summary>
        /// Конструктор входного слоя
        /// </summary>
        public Layer(
            INeuronFactory neuronFactory,
            int totalNeuronCount
            )
        {
            if (neuronFactory == null)
            {
                throw new ArgumentNullException("neuronFactory");
            }

            this.Neurons = new INeuron[totalNeuronCount];

            for (var cc = 0; cc < this.TotalNeuronCount; cc++)
            {
                this.Neurons[cc] = neuronFactory.CreateInputNeuron(cc);
            }
        }

        /// <summary>
        /// Конструктор скрытых и выходного слоя
        /// </summary>
        public Layer(
            INeuronFactory neuronFactory,
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount
            )
        {
            if (neuronFactory == null)
            {
                throw new ArgumentNullException("neuronFactory");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            LayerActivationFunction = activationFunction;

            this.Neurons = new INeuron[currentLayerNeuronCount];

            for (var cc = 0; cc < this.TotalNeuronCount; cc++)
            {
                this.Neurons[cc] = neuronFactory.CreateTrainableNeuron(previousLayerNeuronCount);
            }
        }

        public string GetLayerInformation()
        {
            var neuronCount = this.TotalNeuronCount;

            return
                string.Format(
                    "{0}{1}",
                    neuronCount,
                    this.LayerActivationFunction != null
                        ? this.LayerActivationFunction.ShortName
                        : "Input");
        }

        public ILayerConfiguration GetConfiguration()
        {
            return 
                new LayerConfiguration(
                    this.Neurons.ConvertAll(j => j.GetConfiguration()),
                    this.TotalNeuronCount
                    );
        }
    }
}
