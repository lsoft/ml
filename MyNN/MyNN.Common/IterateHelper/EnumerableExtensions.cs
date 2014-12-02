using System;
using System.Collections.Generic;

namespace MyNN.Common.IterateHelper
{
    public static class EnumerableExtensions
    {
        public static IEnumerable<ZipEntry<T1, T2>> ZipInequalLength<T1, T2>(
            this IEnumerable<T1> collection1,
            IEnumerable<T2> collection2
            )
        {
            if (collection1 == null)
            {
                throw new ArgumentNullException("collection1");
            }
            if (collection2 == null)
            {
                throw new ArgumentNullException("collection2");
            }

            var index = 0;
            using (var enumerator1 = collection1.GetEnumerator())
            {
                using (var enumerator2 = collection2.GetEnumerator())
                {
                    while (enumerator1.MoveNext() && enumerator2.MoveNext())
                    {
                        yield
                            return new ZipEntry<T1, T2>(
                                index,
                                enumerator1.Current,
                                enumerator2.Current
                                );

                        index++;
                    }
                }
            }



        }

        public static IEnumerable<ZipEntry<T1, T2>> ZipEqualLength<T1, T2>(
            this IEnumerable<T1> collection1,
            IEnumerable<T2> collection2
            )
        {
            if (collection1 == null)
            {
                throw new ArgumentNullException("collection1");
            }
            if (collection2 == null)
            {
                throw new ArgumentNullException("collection2");
            }

            var index = 0;
            using (var enumerator1 = collection1.GetEnumerator())
            {
                using (var enumerator2 = collection2.GetEnumerator())
                {
                    while (true)
                    {
                        var hasNext1 = enumerator1.MoveNext();
                        var hasNext2 = enumerator2.MoveNext();

                        if (hasNext1 != hasNext2)
                        {
                            throw new InvalidOperationException(
                                "One of the collections ran out of values before the other");
                        }

                        if (!hasNext1)
                        {
                            break;
                        }

                        yield
                            return new ZipEntry<T1, T2>(
                                index,
                                enumerator1.Current,
                                enumerator2.Current
                                );

                        index++;
                    }
                }
            }



        }
    }
}