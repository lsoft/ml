using System;
using System.Globalization;

namespace MyNN.NeuralNet.Structure.Neurons.Function
{
    [Serializable]
    public class SparsifierFunction : IFunction
    {

        private readonly float _lambda = 2.0f;
        private readonly float _mu = 15.0f;

        public SparsifierFunction()
        {
        }

        public SparsifierFunction(float lambda, float mu)
        {
            _lambda = lambda;
            _mu = mu;
        }

        public float Compute(float x)
        {
            var r = (float) ((1 - Math.Exp(-this._lambda*x))/(1 + Math.Exp(-this._mu*x)));
            return r;
        }

        //http://www.wolframalpha.com/input/?i=first+derivative+%281-exp%28-lambda*x%29%29%2F%281%2Bexp%28-mu*x%29%29
        public float ComputeFirstDerivative(float computed)
        {
            var emuxp1 = Math.Exp(_mu*computed) + 1.0;

            var chislitel = Math.Exp(computed * (_mu - _lambda)) * (_lambda * emuxp1 + _mu * (Math.Exp(_lambda * computed) - 1.0));
            var znamenatel = Math.Pow(emuxp1, 2);

            var fd = (float) (chislitel/znamenatel);

            return fd;
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format(
                    "((1.0 - exp(-1.0 * {0} * {2})) / (1.0 + exp(-1.0 * {1} * {2})))",
                    _lambda.ToString(CultureInfo.InvariantCulture),
                    _mu.ToString(CultureInfo.InvariantCulture),
                    varName);
        }

        public string GetOpenCLFirstDerivative(string varName)
        {
            return
                string.Format(
                    "((exp({2} * ({1} - {0})) * ({0} * (exp({1} * {2}) + 1.0)  + {1} * (exp({0} * {2}) - 1.0)) ) / ((exp({1} * {2}) + 1.0)*(exp({1} * {2}) + 1.0)))",
                    _lambda.ToString(CultureInfo.InvariantCulture),
                    _mu.ToString(CultureInfo.InvariantCulture),
                    varName);
        }
    }
}
