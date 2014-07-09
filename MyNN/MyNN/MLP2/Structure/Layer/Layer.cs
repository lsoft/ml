using System;
using System.Linq;
using MyNN.MLP2.Structure.Neurons;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.Randomizer;

namespace MyNN.MLP2.Structure.Layer
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

        ///// <summary>
        ///// Конструктор скрытых и выходного слоя
        ///// </summary>
        //public Layer(
        //    INeuron[] neuronList,
        //    bool isNeedBiasNeuron)
        //{
        //    if (neuronList == null)
        //    {
        //        throw new ArgumentNullException("neuronList");
        //    }

        //    _isNeedBiasNeuron = isNeedBiasNeuron;
        //    LayerActivationFunction = neuronList[0].ActivationFunction;

        //    var totalNeuronCount = neuronList.Length + (isNeedBiasNeuron ? 1 : 0);

        //    this.Neurons = new INeuron[totalNeuronCount];

        //    for (var cc = 0; cc < this.Neurons.Length; cc++)
        //    {
        //        if (isNeedBiasNeuron && (cc == this.Neurons.Length - 1))
        //        {
        //            this.Neurons[cc] = new BiasNeuron();
        //        }
        //        else
        //        {
        //            this.Neurons[cc] = neuronList[cc];
        //        }
        //    }
        //}

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

            var totalNeuronCount = withoutBiasNeuronCount + 1;//(isNeedBiasNeuron ? 1 : 0);

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

        //извлечь в спец вид фабрики ILayerFromRBMFactory
        /*
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
        //*/
    }
}
