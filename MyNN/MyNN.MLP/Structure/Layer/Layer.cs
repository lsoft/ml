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
        public IDimension SpatialDimension
        {
            get;
            private set;
        }

        public int TotalNeuronCount
        {
            get
            {
                return
                    this.SpatialDimension.TotalNeuronCount;
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
            IDimension spatialDimension
            )
        {
            if (neuronFactory == null)
            {
                throw new ArgumentNullException("neuronFactory");
            }
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }

            SpatialDimension = spatialDimension;

            this.Neurons = new INeuron[TotalNeuronCount];

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
            IDimension spatialDimension,
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
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }

            LayerActivationFunction = activationFunction;
            SpatialDimension = spatialDimension;

            this.Neurons = new INeuron[this.TotalNeuronCount];

            for (var cc = 0; cc < this.TotalNeuronCount; cc++)
            {
                this.Neurons[cc] = neuronFactory.CreateTrainableNeuron(previousLayerNeuronCount);
            }
        }

        public string GetLayerInformation()
        {
            return
                string.Format(
                    "{0}{1}",
                    this.SpatialDimension.GetDimensionInformation(),
                    this.LayerActivationFunction != null
                        ? this.LayerActivationFunction.ShortName
                        : "Input");
        }

        public ILayerConfiguration GetConfiguration()
        {
            return 
                new LayerConfiguration(
                    this.SpatialDimension,
                    this.Neurons.ConvertAll(j => j.GetConfiguration())
                    );
        }
    }
}
