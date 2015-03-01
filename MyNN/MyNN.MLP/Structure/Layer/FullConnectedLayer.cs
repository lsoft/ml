using System;
using System.Linq;
using MyNN.Common.Other;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Layer
{
    public interface IFullConnectedLayer : ILayer
    {
        
    }

    [Serializable]
    public class FullConnectedLayer : IFullConnectedLayer
    {
        public LayerTypeEnum Type
        {
            get;
            private set;
        }

        public IDimension SpatialDimension
        {
            get;
            private set;
        }

        /// <summary>
        /// Всего нейронов в слое
        /// </summary>
        public int TotalNeuronCount
        {
            get
            {
                return
                    this.SpatialDimension.Multiplied;
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
        public FullConnectedLayer(
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

            this.Type = LayerTypeEnum.Input;
            this.SpatialDimension = spatialDimension;

            this.Neurons = new INeuron[TotalNeuronCount];

            for (var cc = 0; cc < this.TotalNeuronCount; cc++)
            {
                this.Neurons[cc] = neuronFactory.CreateInputNeuron(cc);
            }
        }

        /// <summary>
        /// Конструктор скрытых и выходного слоя
        /// </summary>
        public FullConnectedLayer(
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

            this.Type = LayerTypeEnum.FullConnected;
            this.LayerActivationFunction = activationFunction;
            this.SpatialDimension = spatialDimension;

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
                    "FC({0} {1})",
                    this.SpatialDimension.GetDimensionInformation(),
                    this.LayerActivationFunction != null
                        ? this.LayerActivationFunction.ShortName
                        : "Input");
        }

        public ILayerConfiguration GetConfiguration()
        {
            return 
                new LayerConfiguration(
                    this.LayerActivationFunction,
                    this.SpatialDimension,
                    this.Neurons.Sum(j => j.Weights.Length),
                    this.TotalNeuronCount,
                    this.Neurons.ConvertAll(j => j.GetConfiguration())
                    );
        }

        /// <summary>
        /// Получить массив клонированных весов всех нейронов сети
        /// </summary>
        public void GetClonedWeights(
            out float[] weights,
            out float[] biases
            )
        {
            weights = new float[this.Neurons.Sum(j => j.Weights.Length)];
            biases = new float[this.TotalNeuronCount];

            var weightShift = 0;

            for (var neuronIndex = 0; neuronIndex < this.TotalNeuronCount; neuronIndex++)
            {
                var neuron = this.Neurons[neuronIndex];

                Array.Copy(
                    neuron.Weights,
                    0,
                    weights,
                    weightShift,
                    neuron.Weights.Length);

                weightShift += neuron.Weights.Length;

                biases[neuronIndex] = neuron.Bias;
            }


        }

        /// <summary>
        /// Записать веса в слой
        /// </summary>
        public void SetWeights(
            float[] weights,
            float[] biases
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (biases == null)
            {
                throw new ArgumentNullException("biases");
            }

            var weightShiftIndex = 0;
            for (var neuronIndex = 0; neuronIndex < this.TotalNeuronCount; ++neuronIndex)
            {
                var neuron = this.Neurons[neuronIndex];

                var weightCount = neuron.Weights.Length;

                Array.Copy(weights, weightShiftIndex, neuron.Weights, 0, weightCount);
                weightShiftIndex += weightCount;

                neuron.Bias = biases[neuronIndex];
            }
        }
    }
}
