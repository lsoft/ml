using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN
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
