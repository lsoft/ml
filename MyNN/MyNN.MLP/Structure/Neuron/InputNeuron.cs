using System;
using MyNN.Common.OpenCLHelper;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Structure.Neuron
{
    [Serializable]
    public class InputNeuron : INeuron
    {
        private readonly int _thisIndex;

        #region private class

        [Serializable]
        private class InputConstFunction : IFunction
        {
            public string ShortName
            {
                get
                {
                    return
                        "ICF1";
                }
            }

            public float Compute(float x)
            {
                return 1f;
            }

            public float ComputeFirstDerivative(float x)
            {
                throw new NotSupportedException("Для этой функции этот метод не должен быть вызван");
            }

            public string GetOpenCLFirstDerivative(string varName)
            {
                throw new NotSupportedException("Для этой функции этот метод не должен быть вызван");
            }

            public string GetOpenCLActivationFunction(string varName)
            {
                return
                    "(1.0)";
            }

            public string GetOpenCLActivationMethod(
                string methodName,
                VectorizationSizeEnum vse
                )
            {
                throw new NotSupportedException("Для этой функции этот метод не должен быть вызван");
            }

            public string GetOpenCLDerivativeMethod(
                string methodName,
                VectorizationSizeEnum vse
                )
            {
                throw new NotSupportedException("Для этой функции этот метод не должен быть вызван");
            }
        }

        #endregion

        public float[] Weights
        {
            get;
            private set;
        }

        public IFunction ActivationFunction
        {
            get;
            private set;
        }

        public bool IsBiasNeuron
        {
            get
            {
                return false;
            }
        }

        public INeuronConfiguration GetConfiguration()
        {
            return 
                new NeuronConfiguration(
                    this.Weights.Length,
                    this.IsBiasNeuron);
        }

        public InputNeuron(
            int thisIndex)
        {
            this._thisIndex = thisIndex;
            this.ActivationFunction = new InputConstFunction();

            //случайные веса
            this.Weights = new float[0];
        }
    }
}
