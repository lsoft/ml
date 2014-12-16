using System;
using System.Collections.Generic;
using System.Linq;

namespace MyNN.Common.Other
{
    public static class SplitHelper
    {
        public static IEnumerable<List<T>> LazySplit<T>(
            this IEnumerable<T> list,
            int splitCount
            )
        {
            if (splitCount <= 0)
            {
                throw new ArgumentException("splitCount <= 0");
            }

            var result = new List<T>();

            var currentIndex = 0;
            foreach (var i in list)
            {
                result.Add(i);

                if (++currentIndex >= splitCount)
                {
                    yield return
                        result;

                    result = new List<T>();
                    currentIndex = 0;
                }
            }

            if (result.Count > 0)
            {
                yield return result;
            }
        }

        public static List<List<T>> CompleteSplit<T>(
            this IEnumerable<T> list,
            int splitCount
            )
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
