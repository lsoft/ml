using System;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Neurons
{
    [Serializable]
    public class BiasMLPNeuron : 
        TrainableMLPNeuron
    {
        [Serializable]
        private class BiasConstFunction : IFunction
        {
            public string ShortName
            {
                get
                {
                    return
                        "BCF1";
                }
            }

            public float Compute(float x)
            {
                return 1f;
            }

            public float ComputeFirstDerivative(float x)
            {
                throw new InvalidOperationException("Неприменимо");
            }

            public string GetOpenCLFirstDerivative(string varName)
            {
                throw new InvalidOperationException("Неприменимо");
            }

            public string GetOpenCLActivationFunction(string varName)
            {
                return
                    "(1.0)";
            }
        }

        public BiasMLPNeuron()
        {
            this.ActivationFunction = new BiasConstFunction();

            //случайные веса
            this.Weights = new float[0];
        }

        public override bool IsBiasNeuron
        {
            get
            {
                return true;
            }
        }
    }
}
