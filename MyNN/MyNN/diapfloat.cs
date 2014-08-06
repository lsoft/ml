using System;

namespace MyNN
{
    // ReSharper disable once InconsistentNaming
    public class diapfloat
    {
        private readonly float _center;

        public float Min
        {
            get;
            private set;
        }

        public float Max
        {
            get;
            private set;
        }

        public diapfloat(float min, float max, bool nevermind)
        {
            _center = min + (max - min)/2f;

            Min = min;
            Max = max;

        }

        public diapfloat(float center, float allowedError)
        {
            if (allowedError < 0f)
            {
                throw new ArgumentException("allowedError < 0f");
            }

            _center = center;

            Min = center - allowedError;
            Max = center + allowedError;

        }

        public bool IsValueInclusive(float testValue)
        {
            return
                testValue >= Min
                && testValue <= Max;
        }

        public override string ToString()
        {
            var result = string.Format(
                "[{0} <-> {1}]",
                Min,
                Max
                );

            return result;
        }
    }
}