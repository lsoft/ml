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
        private readonly INeuronFactory _neuronFactory;
        private bool _isNeedBiasNeuron;

        public bool IsBiasNeuronExists
        {
            get
            {
                return
                    _isNeedBiasNeuron;
            }
        }

        public int NonBiasNeuronCount
        {
            get
            {
                var result = this.Neurons.Length;

                if (_isNeedBiasNeuron)
                {
                    result--;
                }

                return result;
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
            int withoutBiasNeuronCount)
        {
            if (neuronFactory == null)
            {
                throw new ArgumentNullException("neuronFactory");
            }

            _neuronFactory = neuronFactory;
            
            _isNeedBiasNeuron = true;

            var totalNeuronCount = withoutBiasNeuronCount + 1;

            this.Neurons = new INeuron[totalNeuronCount];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                this.Neurons[cc] = neuronFactory.CreateInputNeuron(cc);

                if (cc == this.Neurons.Length - 1)
                {
                    this.Neurons[cc] = neuronFactory.CreateBiasNeuron();
                }
            }
        }

        /// <summary>
        /// Конструктор скрытых и выходного слоя
        /// </summary>
        public Layer(
            INeuronFactory neuronFactory,
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount,
            bool isNeedBiasNeuron,
            bool isPreviousLayerHadBiasNeuron)
        {
            if (neuronFactory == null)
            {
                throw new ArgumentNullException("neuronFactory");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            _neuronFactory = neuronFactory;
            _isNeedBiasNeuron = isNeedBiasNeuron;
            LayerActivationFunction = activationFunction;

            var totalNeuronCount = currentLayerNeuronCount + (isNeedBiasNeuron ? 1 : 0);

            this.Neurons = new INeuron[totalNeuronCount];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                this.Neurons[cc] = neuronFactory.CreateTrainableNeuron(
                    activationFunction,
                    previousLayerNeuronCount + (isPreviousLayerHadBiasNeuron ? 1 : 0));

                if (isNeedBiasNeuron && (cc == this.Neurons.Length - 1))
                {
                    this.Neurons[cc] = neuronFactory.CreateBiasNeuron();
                }
            }
        }

        public void AddBiasNeuron()
        {
            var nln = this.Neurons.Last();
            if (nln.IsBiasNeuron)
            {
                throw new Exception("Уже добавлен биас нейрон");
            }

            var na = new INeuron[this.NonBiasNeuronCount + 1];
            Array.Copy(this.Neurons, na, this.NonBiasNeuronCount);
            na[na.Length - 1] = _neuronFactory.CreateBiasNeuron();

            this.Neurons = na;
            this._isNeedBiasNeuron = true;
        }

        public void RemoveBiasNeuron()
        {
            var nln = this.Neurons.Last();
            if (nln.IsBiasNeuron)
            {
                var na = new INeuron[this.NonBiasNeuronCount];
                Array.Copy(this.Neurons, na, this.NonBiasNeuronCount);

                this.Neurons = na;
                this._isNeedBiasNeuron = false;
            }
        }

        public string GetLayerInformation()
        {
            var neuronCount = this.Neurons.Length;
            var nonbiasNeuronCount = this.NonBiasNeuronCount;
            var biasNeuronCount = neuronCount - nonbiasNeuronCount;

            return
                string.Format(
                    "{0}({1}+{2}){3}",
                    neuronCount,
                    nonbiasNeuronCount,
                    biasNeuronCount,
                    this.LayerActivationFunction != null
                        ? this.LayerActivationFunction.ShortName
                        : "Input");
        }

        public ILayerConfiguration GetConfiguration()
        {
            return 
                new LayerConfiguration(
                    this.Neurons.ConvertAll(j => j.GetConfiguration()),
                    this.IsBiasNeuronExists,
                    this.NonBiasNeuronCount
                    );
        }
    }
}
