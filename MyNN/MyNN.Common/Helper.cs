using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.Common
{
    public static class Helper
    {
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
