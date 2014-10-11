using System;
using MyNN.Common.OpenCLHelper;

namespace MyNN.MLP.Structure.Neuron.Function
{
    [Serializable]
    public class DRLUFunction : IFunction
    {

        public DRLUFunction()
        {
        }

        public string ShortName
        {
            get
            {
                return "IRLU";
            }
        }

        public float Compute(float x)
        {
            return
                Math.Max(0, x);
        }

        public float ComputeFirstDerivative(float computed)
        {
            return 1f;
        }

        public string GetOpenCLActivationFunction(string varName)
        {
            return
                string.Format(
                    "max((float)(0.0), {0})",
                    varName);
        }

        public string GetOpenCLFirstDerivative(string varName)
        {
            return
                "(1.0)";
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
    const floatv zero = 0.0;

    floatv result = max(zero, incoming);

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
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }


    }
}
