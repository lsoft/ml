using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN
{
    public class KahanAlgorithm
    {
        public static float Reduce(float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            if (data.Length == 0)
            {
                return 0;
            }

            var sum = data[0];
            var c = 0.0f;
            for (var i = 1; i < data.Length; i++)
            {
                var y = data[i] - c;
                var t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }

            return sum;
        }

    }
}
