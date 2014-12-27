using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.Common
{
    public static class Helper
    {
        public static void Foreach<T>(this T[] list, Action<T> method)
        {
            if (list == null)
            {
                throw new ArgumentNullException("list");
            }
            if (method == null)
            {
                throw new ArgumentNullException("method");
            }

            for (var cc = 0; cc < list.Length; cc++)
            {
                method(list[cc]);
            }
        }

        public static int UpTo(int value, int step)
        {
            if (value < step)
            {
                return step;
            }

            var ostatok = value % step;

            if (ostatok == 0)
            {
                return value;
            }

            return
                value + step - ostatok;

        }

    }
}
