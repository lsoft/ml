using System;
using System.Linq;
using MyNN.NeuralNet.Structure.Neurons;
using MyNN.NeuralNet.Structure.Neurons.Function;

namespace MyNN.NeuralNet.Structure.Layers
{
    [Serializable]
    public class Layer
        //: ILayer,
        //ITrainLayer<TrainableNeuron>
    {
        private bool _isNeedBiasNeuron;

        public float[] LastOutput
        {
            get;
            private set;
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

        public TrainableNeuron[] Neurons
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
            int currentLayerNeuronCount,
            bool isNeedBiasNeuron)
        {
            _isNeedBiasNeuron = isNeedBiasNeuron;

            var totalNeuronCount = currentLayerNeuronCount + (isNeedBiasNeuron ? 1 : 0);

            this.Neurons = new TrainableNeuron[totalNeuronCount];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                this.Neurons[cc] = new InputNeuron(
                    new SigmoidFunction(1),
                    cc);

                if (isNeedBiasNeuron && (cc == this.Neurons.Length - 1))
                {
                    this.Neurons[cc] = new BiasNeuron(new SigmoidFunction(1));
                }
            }
        }

        /// <summary>
        /// Конструктор скрытых и выходного слоя
        /// </summary>
        public Layer(
            int currentLayerNeuronCount ,
            int previousLayerNeuronCount,
            bool isNeedBiasNeuron,
            bool isPreviousLayerHadBiasNeuron,
            ref int seed)
            : this(
                new SigmoidFunction(1),
                currentLayerNeuronCount,
                previousLayerNeuronCount,
                isNeedBiasNeuron,
                isPreviousLayerHadBiasNeuron,
                ref seed)
        {
            
        }

        /// <summary>
        /// Конструктор скрытых и выходного слоя
        /// </summary>
        public Layer(
            IFunction activationFunction,
            int currentLayerNeuronCount,
            int previousLayerNeuronCount,
            bool isNeedBiasNeuron,
            bool isPreviousLayerHadBiasNeuron,
            ref int seed)
        {
            _isNeedBiasNeuron = isNeedBiasNeuron;
            LayerActivationFunction = activationFunction;

            var totalNeuronCount = currentLayerNeuronCount + (isNeedBiasNeuron ? 1 : 0);

            this.Neurons = new TrainableNeuron[totalNeuronCount];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                this.Neurons[cc] = new HiddeonOutputNeuron(
                    activationFunction,
                    previousLayerNeuronCount + (isPreviousLayerHadBiasNeuron ? 1 : 0),
                    ref seed);

                if (isNeedBiasNeuron && (cc == this.Neurons.Length - 1))
                {
                    this.Neurons[cc] = new BiasNeuron(new SigmoidFunction(1));
                }
            }
        }


        /// <summary>
        /// Конструктор скрытых и выходного слоя
        /// </summary>
        public Layer(
            TrainableNeuron[] neuronList,
            bool isNeedBiasNeuron)
        {
            if (neuronList == null)
            {
                throw new ArgumentNullException("neuronList");
            }

            _isNeedBiasNeuron = isNeedBiasNeuron;
            LayerActivationFunction = neuronList[0].ActivationFunction;

            var totalNeuronCount = neuronList.Length + (isNeedBiasNeuron ? 1 : 0);

            this.Neurons = new TrainableNeuron[totalNeuronCount];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                if (isNeedBiasNeuron && (cc == this.Neurons.Length - 1))
                {
                    this.Neurons[cc] = new BiasNeuron(new SigmoidFunction(1));
                }
                else
                {
                    this.Neurons[cc] = neuronList[cc];
                }
            }
        }

        public void RemoveBiasNeuron()
        {
            var nln = this.Neurons.Last();
            if (nln is BiasNeuron)
            {
                var na = new TrainableNeuron[this.NonBiasNeuronCount];
                Array.Copy(this.Neurons, na, this.NonBiasNeuronCount);

                this.Neurons = na;
                _isNeedBiasNeuron = false;
            }
        }

        public float[] Compute(float[] inputVector)
        {
            this.LastOutput = new float[this.Neurons.Length];

            for (var cc = 0; cc < this.Neurons.Length; cc++)
            {
                this.LastOutput[cc] = this.Neurons[cc].Activate(inputVector);
            }

            return
                this.LastOutput;
        }


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
                    if (neuron is HiddeonOutputNeuron)
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

                if (neuron is HiddeonOutputNeuron)
                {
                    for (var weightIndex = 0; weightIndex < neuron.Weights.Length; weightIndex++)
                    {
                        var weight = weightFile[weightIndex * (this.NonBiasNeuronCount + 1) + neuronIndex];
                        neuron.Weights[weightIndex] = weight;
                    }
                }
            }
        }

        public string DumpLayerInformation()
        {
            var neuronCount = this.Neurons.Length;
            var nonbiasNeuronCount = this.NonBiasNeuronCount;
            var biasNeuronCount = neuronCount - nonbiasNeuronCount;

            return
                string.Format(
                    "{0}({1}+{2})",
                    neuronCount,
                    nonbiasNeuronCount,
                    biasNeuronCount);
        }

    }
}
