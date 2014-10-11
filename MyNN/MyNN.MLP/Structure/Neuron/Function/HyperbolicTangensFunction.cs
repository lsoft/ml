using System;
using System.Globalization;
using MyNN.Common.OpenCLHelper;

namespace MyNN.MLP.Structure.Neuron.Function
{
    [Serializable]
    public class HyperbolicTangensFunction : IFunction
    {

        private readonly float _alpha = 0.0f;
        private readonly float _beta = 0.0f;

        public string ShortName
        {
            get
            {
                return "HTan";
            }
        }

        public HyperbolicTangensFunction()
            : this(1.7159f, 0.6666f)
        {
        }

        public HyperbolicTangensFunction(float alpha, float beta)
        {
            _alpha = alpha;
            _beta = beta;
        }

        public float Compute(float x)
        {
            return 
                (float)(_alpha * Math.Tanh(_beta * x));
        }

        public float ComputeFirstDerivative(float x)
        {
            var computed = this.Compute(x);

            var scaled = computed/_alpha;

            return
                _alpha * _beta * (1f - scaled * scaled);
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format("((float)({0}) * tanh((float)({1}) * {2}) )",
                      _alpha.ToString(CultureInfo.InvariantCulture),
                      _beta.ToString(CultureInfo.InvariantCulture),
                      varName);
        }

        public string GetOpenCLFirstDerivative(string varName)
        {
            var computed = this.GetOpenCLActivationFunction(varName);

            return
                string.Format(@"
({0} * {1} * (((float)1) - ({2} / {0}) * ({2} / {0})))
",
                      _alpha.ToString(CultureInfo.InvariantCulture),
                      _beta.ToString(CultureInfo.InvariantCulture),
                      computed);

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
    const floatv alpha = {ALPHA};
    const floatv beta = {BETA};

    floatv result = alpha * tanh(beta * incoming);

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
                "{ALPHA}",
                _alpha.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{BETA}",
                _beta.ToString(CultureInfo.InvariantCulture)
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
{ACTIVATION_METHOD}

inline floatv {METHOD_NAME}(floatv incoming)
{
    const floatv one = 1.0;
    const floatv alpha = {ALPHA};
    const floatv beta = {BETA};

    floatv activation = {ACTIVATION_METHOD_NAME}(incoming);

    floatv result = alpha * beta * (one - (activation / alpha) * (activation / alpha));

    return result;
}
";


//            var computed = this.GetOpenCLActivationFunction(varName);
//            return
//                string.Format(@"
//{0} * {1} * (1 - ({2} / {0}) * ({2} / {0}))
//",
//                      _alpha.ToString(CultureInfo.InvariantCulture),
//                      _beta.ToString(CultureInfo.InvariantCulture),
//                      computed);

            var activationKernelName =
                string.Format(
                    "ActivationKernel{0}",
                    Guid.NewGuid());

            activationKernelName = activationKernelName.Replace("-", "");

            var activationKernel = this.GetOpenCLActivationMethod(
                activationKernelName,
                vse);


            var vsize = VectorizationHelper.GetVectorizationSuffix(vse);

            var result = methodBody;

            result = result.Replace(
                "{ACTIVATION_METHOD}",
                activationKernel
                );

            result = result.Replace(
                "{ACTIVATION_METHOD_NAME}",
                activationKernelName
                );

            result = result.Replace(
                "floatv",
                string.Format(
                    "float{0}",
                    vsize));

            result = result.Replace(
                "{ALPHA}",
                _alpha.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{BETA}",
                _beta.ToString(CultureInfo.InvariantCulture)
                );

            result = result.Replace(
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }
    }

}
