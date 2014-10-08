using System;
using System.Globalization;

namespace MyNN.MLP.Structure.Neuron.Function
{
    [Serializable]
    public class LinearFunction : IFunction
    {

        private readonly float _alpha = 1.0f;

        public string ShortName
        {
            get
            {
                return "Lin";
            }
        }

        public LinearFunction(float alpha)
        {
            _alpha = alpha;
        }

        public float Compute(float x)
        {
            var r = _alpha * x;
            return r;
        }

        public float ComputeFirstDerivative(float computed)
        {
            return
                _alpha;
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format(
                    "({0} * {1})",
                    _alpha.ToString(CultureInfo.InvariantCulture),
                    varName);
        }

        public string GetOpenCLFirstDerivative(string varName)
        {
            return
                string.Format(
                    "({0})",
                    _alpha.ToString(CultureInfo.InvariantCulture));
        }
    }
}
