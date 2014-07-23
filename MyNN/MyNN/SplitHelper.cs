using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN
{
    public static class SplitHelper
    {
        public static List<List<T>> Split<T>(this IEnumerable<T> list, int splitCount)
        {
            if (splitCount <= 0)
            {
                throw new ArgumentException("splitCount <= 0");
            }

            var result = new List<List<T>>();
            var from = new List<T>(list);

            while (from.Count > 0)
            {
                if (from.Count > splitCount)
                {
                    var part = from.Take(splitCount).ToList();

                    result.Add(part);

                    from.RemoveRange(0, splitCount);
                }
                else
                {
                    result.Add(from);
                    break;
                }
            }

            return result;
        }
    }
}
