using System;
using System.Linq;
using MyNN.MLP2.Structure.Neurons;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.Randomizer;


namespace MyNN.MLP2.Structure
{
    [Serializable]
    public class MLPLayer
    {
        private bool _isNeedBiasNeuron;
        private readonly IRandomizer _randomizer;

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

        public TrainableMLPNeuron[] Neurons
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
        /// Конструктор скрытых и выходного слоя
        /// </summary>
        public MLPLayer(
            TrainableMLPNeuron[] neuronList,
            bool isNeedBiasNeuron)
        {
            if (neuronList == null)
            {
                throw new ArgumentNullException("neuronList");
            }

            _isNeedBiasNeuron = isNeedBiasNeuron;
            LayerActivationFunction = neuronList[0].ActivationFunction;

            var totalNeuronCount = neuronList.Length + (isNeedBiasNeuron ? 1 : 0);

            this.Neurons = new TrainableMLPNeuron[totalNeuronCount];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                if (isNeedBiasNeuron && (cc == this.Neurons.Length - 1))
                {
                    this.Neurons[cc] = new BiasMLPNeuron();
                }
                else
                {
                    this.Neurons[cc] = neuronList[cc];
                }
            }
        }

        /// <summary>
        /// Конструктор входного слоя
        /// </summary>
        public MLPLayer(
            int currentLayerNeuronCount,
            bool isNeedBiasNeuron)
        {
            _isNeedBiasNeuron = isNeedBiasNeuron;

            var totalNeuronCount = currentLayerNeuronCount + (isNeedBiasNeuron ? 1 : 0);

            this.Neurons = new TrainableMLPNeuron[totalNeuronCount];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                this.Neurons[cc] = new InputMLPNeuron(
                    cc);

                if (isNeedBiasNeuron && (cc == this.Neurons.Length - 1))
                {
                    this.Neurons[cc] = new BiasMLPNeuron();
                }
            }
        }

        /// <summary>
        /// Конструктор скрытых и выходного слоя
        /// </summary>
        public MLPLayer(
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount,
            bool isNeedBiasNeuron,
            bool isPreviousLayerHadBiasNeuron,
            IRandomizer randomizer)
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }

            _isNeedBiasNeuron = isNeedBiasNeuron;
            _randomizer = randomizer;
            LayerActivationFunction = activationFunction;

            var totalNeuronCount = currentLayerNeuronCount + (isNeedBiasNeuron ? 1 : 0);

            this.Neurons = new TrainableMLPNeuron[totalNeuronCount];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                this.Neurons[cc] = new HiddeonOutputMLPNeuron(
                    activationFunction,
                    previousLayerNeuronCount + (isPreviousLayerHadBiasNeuron ? 1 : 0),
                    _randomizer);

                if (isNeedBiasNeuron && (cc == this.Neurons.Length - 1))
                {
                    this.Neurons[cc] = new BiasMLPNeuron();
                }
            }
        }

        public void AddBiasNeuron()
        {
            var nln = this.Neurons.Last();
            if (nln is BiasMLPNeuron)
            {
                throw new Exception("Уже добавлен биас нейрон");
            }

            var na = new TrainableMLPNeuron[this.NonBiasNeuronCount + 1];
            Array.Copy(this.Neurons, na, this.NonBiasNeuronCount);
            na[na.Length - 1] = new BiasMLPNeuron();

            this.Neurons = na;
            this._isNeedBiasNeuron = true;
        }

        public void RemoveBiasNeuron()
        {
            var nln = this.Neurons.Last();
            if (nln is BiasMLPNeuron)
            {
                var na = new TrainableMLPNeuron[this.NonBiasNeuronCount];
                Array.Copy(this.Neurons, na, this.NonBiasNeuronCount);

                this.Neurons = na;
                this._isNeedBiasNeuron = false;
            }
        }

        public string DumpLayerInformation()
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

        #region deep belief network related code

        /// <summary>
        /// Загружаем веса из предобученной rbm
        /// </summary>
        /// <param name="pathToWeightsFile">Путь к файлу weights.bin</param>
        public void LoadWeightsFromRBM(
            string pathToWeightsFile)
        {
            var weightFile = SerializationHelper.LoadFromFile<float[]>(pathToWeightsFile);
            int lineIndex = 0;
            for (int neuronIndex = 0; neuronIndex < this.Neurons.Length; neuronIndex++)
            {
                var neuron = this.Neurons[neuronIndex];

                for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++, lineIndex++)
                {
                    if (neuron is HiddeonOutputMLPNeuron)
                    {
                        var weight = weightFile[lineIndex];
                        neuron.Weights[weightIndex] = weight;
                    }
                }
            }
        }

        /// <summary>
        /// Загружаем веса из предобученной rbm в режиме автоенкодера ("наоборот")
        /// </summary>
        /// <param name="pathToWeightsFile">Путь к файлу weights.bin</param>
        public void LoadAutoencoderWeightsFromRBM(
            string pathToWeightsFile)
        {
            var weightFile = SerializationHelper.LoadFromFile<float[]>(pathToWeightsFile);

            for (int neuronIndex = 0; neuronIndex < this.Neurons.Length; neuronIndex++)
            {
                var neuron = this.Neurons[neuronIndex];

                if (neuron is HiddeonOutputMLPNeuron)
                {
                    for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++)
                    {
                        var weight = weightFile[weightIndex * (this.NonBiasNeuronCount + 1) + neuronIndex];
                        neuron.Weights[weightIndex] = weight;
                    }
                }
            }
        }

        #endregion

    }
}
