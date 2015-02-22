using System;

namespace MyNN.MLP.Structure.Neuron
{
    public class PseudoNeuron : INeuron
    {
        public float[] Weights
        {
            get
            {
                throw new InvalidOperationException("В псевдонейроне запрос весов напрямую из нейрона не допускается");
            }
        }

        public float Bias
        {
            get
            {
                throw new InvalidOperationException("В псевдонейроне запрос биаса напрямую из нейрона не допускается");
            }

            set
            {
                throw new InvalidOperationException("В псевдонейроне запрос на изменение биаса напрямую из нейрона не допускается");
            }
        }

        public PseudoNeuron(
            )
        {
        }

        public INeuronConfiguration GetConfiguration()
        {
            return
                new NeuronConfiguration(
                    0
                    );

        }
    }
}