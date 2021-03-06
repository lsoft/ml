﻿using System;
using System.Globalization;
using MyNN.Common.OpenCLHelper;

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

    floatv result = alpha * incoming;

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

            if (methodName == null)
            {
                throw new ArgumentNullException("methodName");
            }

            const string methodBody = @"
inline floatv {METHOD_NAME}(floatv incoming)
{
    const floatv result = {ALPHA};
            
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
                "{METHOD_NAME}",
                methodName
                );

            return result;
        }
    }
}
