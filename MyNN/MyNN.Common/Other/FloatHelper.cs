using System;

namespace MyNN.Common.Other
{
    public static class FloatHelper
    {
        public static bool IsEquals(
            this float a,
            float b,
            float epsilon)
        {
            var abs = Math.Abs(a - b);

            return
                abs <= epsilon;
        }
    }
}
