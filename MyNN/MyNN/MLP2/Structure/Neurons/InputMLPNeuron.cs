using System;
using MyNN.MLP2.Structure.Neurons.Function;

namespace MyNN.MLP2.Structure.Neurons
{
    [Serializable]
    public class InputMLPNeuron : 
        TrainableMLPNeuron
    {
        private readonly int _thisIndex;

        [Serializable]
        private class InputFunction : IFunction
        {
            public string ShortName
            {
                get
                {
                    return
                        "Input";
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

        public InputMLPNeuron(
            int thisIndex)
        {
            this._thisIndex = thisIndex;
            this.ActivationFunction = new InputFunction();

            //случайные веса
            this.Weights = new float[0];
        }

        public override bool IsBiasNeuron
        {
            get
            {
                return false;
            }
        }
    }
}
