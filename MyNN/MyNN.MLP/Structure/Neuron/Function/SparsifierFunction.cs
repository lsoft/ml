using System;
using System.Globalization;
using MyNN.Common.OpenCLHelper;

namespace MyNN.MLP.Structure.Neuron.Function
{
    [Serializable]
    public class SparsifierFunction : IFunction
    {

        private readonly float _lambda = 2.0f;
        private readonly float _mu = 15.0f;

        public string ShortName
        {
            get
            {
                return "Spars";
            }
        }

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
            var emuxp1 = (float)(Math.Exp(_mu*computed) + 1.0f);

            var chislitel = (float)(Math.Exp(computed * (_mu - _lambda)) * (_lambda * emuxp1 + _mu * (Math.Exp(_lambda * computed) - 1.0)));
            var znamenatel = (float)(Math.Pow(emuxp1, 2));

            var fd = (float) (chislitel/znamenatel);

            return fd;
        }

        public string GetOpenCLActivationMethod(
            string methodName, 
            VectorizationSizeEnum vse
            )
        {
            if (methodName == null)
            {
                throw new ArgumentNullException("methodName");
            }

            const string methodBody = @"
inline floatv {METHOD_NAME}(floatv incoming)
{
    const floatv one = 1.0;
    const floatv mone = -1.0;
    const floatv lamda = {LAMBDA};
    const floatv mu = {MU};

    floatv mul0 = mone * lamda * incoming;
    floatv mul1 = mone * mu * incoming;

    floatv result = (one - exp(mul0)) / (one + exp(mul1));

    return result;
}
";

            var vsize = VectorizationHelper.GetVectorizationSuffix(vse);

            var result = methodBody;

            result = result.Replace(
                "floatv",
                string.Format(
                    "float{0}",
                    vsize));

            result = result.Replace(
                "{LAMBDA}",
                _lambda.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{MU}",
                _mu.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }

        public string GetOpenCLDerivativeMethod(
            string methodName,
            VectorizationSizeEnum vse
            )
        {
            if (methodName == null)
            {
                throw new ArgumentNullException("methodName");
            }

            const string methodBody = @"
inline floatv {METHOD_NAME}(floatv incoming)
{
    const floatv one = 1.0;
    const floatv lamda = {LAMBDA};
    const floatv mu = {MU};

    floatv emi = exp(mu * incoming);
    floatv emi1 = emi + one;
    floatv emi1sq = emi1 * emi1;

    floatv eli = exp(lamda * incoming);

    floatv result = (exp(incoming * (mu - lamda)) * (lamda * (emi + one)  + mu * (eli - one)) ) / emi1sq;

    return result;
}
";

            var vsize = VectorizationHelper.GetVectorizationSuffix(vse);

            var result = methodBody;

            result = result.Replace(
                "floatv",
                string.Format(
                    "float{0}",
                    vsize));

            result = result.Replace(
                "{LAMBDA}",
                _lambda.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{MU}",
                _mu.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }

    }
}
