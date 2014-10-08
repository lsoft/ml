using System;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.CSharp
{
    public class CSharpLayerContainer : ICSharpLayerContainer
    {
        private readonly int _previousLayerTotalNeuronCount;
        private readonly int _currentLayerNonBiasNeuronCount;
        private readonly int _currentLayerTotalNeuronCount;

        public float[] WeightMem
        {
            get;
            private set;
        }

        public float[] NetMem
        {
            get;
            private set;
        }

        public float[] StateMem
        {
            get;
            private set;
        }

        public CSharpLayerContainer(
            int currentLayerNonBiasNeuronCount,
            int currentLayerTotalNeuronCount
            )
        {
            _previousLayerTotalNeuronCount = 0;
            _currentLayerNonBiasNeuronCount = currentLayerNonBiasNeuronCount;
            _currentLayerTotalNeuronCount = currentLayerTotalNeuronCount;

            //нейроны
            NetMem = new float[currentLayerTotalNeuronCount];
            StateMem = new float[currentLayerTotalNeuronCount];
        }

        public CSharpLayerContainer(
            int previousLayerTotalNeuronCount,
            int currentLayerNonBiasNeuronCount,
            int currentLayerTotalNeuronCount
            )
        {
            if (previousLayerTotalNeuronCount == 0)
            {
                throw new ArgumentException("For input layer use another constructor");
            }

            _previousLayerTotalNeuronCount = previousLayerTotalNeuronCount;
            _currentLayerNonBiasNeuronCount = currentLayerNonBiasNeuronCount;
            _currentLayerTotalNeuronCount = currentLayerTotalNeuronCount;

            //нейроны
            NetMem = new float[currentLayerTotalNeuronCount];
            StateMem = new float[currentLayerTotalNeuronCount];

            //веса
            WeightMem = new float[currentLayerTotalNeuronCount*previousLayerTotalNeuronCount];
        }

        public void ClearAndPushHiddenLayers()
        {
            var nml = this.NetMem.Length;
            Array.Clear(this.NetMem, 0, nml);
            this.NetMem[nml - 1] = 1f;

            var sml = this.StateMem.Length;
            Array.Clear(this.StateMem, 0, sml);
            this.StateMem[sml - 1] = 1f;
        }

        public void PushInput(float[] data)
        {
            if (data.Length != _currentLayerNonBiasNeuronCount)
            {
                throw new ArgumentException("data.Length != _currentLayerNonBiasNeuronCount");
            }

            //записываем значения из сети в объекты OpenCL
            for (var neuronIndex = 0; neuronIndex < _currentLayerTotalNeuronCount; neuronIndex++)
            {
                var isBiasNeuron = neuronIndex == _currentLayerNonBiasNeuronCount;

                this.NetMem[neuronIndex] = 0; //LastNET
                this.StateMem[neuronIndex] =
                    isBiasNeuron
                        ? 1.0f
                        : data[neuronIndex];
            }
        }

        public void PushWeights(ILayer layer)
        {
            if (layer == null)
            {
                throw new ArgumentNullException("layer");
            }

            var weightShift = 0;

            var weightMem = this.WeightMem;
            for (var neuronIndex = 0; neuronIndex < layer.NonBiasNeuronCount; neuronIndex++)
            {
                var neuron = layer.Neurons[neuronIndex];

                Array.Copy(
                    neuron.Weights,
                    0,
                    weightMem,
                    weightShift,
                    neuron.Weights.Length);

                weightShift += neuron.Weights.Length;
            }
        }

        public void PopHiddenState()
        {
            //nothing to do
        }

        public void PopLastLayerState()
        {
            //nothing to do
        }

        public ILayerState GetLayerState()
        {
            var ls = new LayerState(
                this.StateMem,
                _currentLayerNonBiasNeuronCount);

            return ls;
        }
    }
}
