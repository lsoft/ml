using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN
{
    public static class ArrayOperations
    {
        public static void Clear<T>(this T[] a)
        {
            for (var cc = 0; cc < a.Length; cc++)
            {
                a[cc] = default(T);
            }
        }

        public static void Fill<T>(this T[] a, T value)
        {
            for (var cc = 0; cc < a.Length; cc++)
            {
                a[cc] = value;
            }
        }

        public static void Fill<T>(this T[] a, Func<T> value)
        {
            for (var cc = 0; cc < a.Length; cc++)
            {
                a[cc] = value();
            }
        }

        public static void Transform<T>(this T[] a, Func<T, T> value)
        {
            for (var cc = 0; cc < a.Length; cc++)
            {
                a[cc] = value(a[cc]);
            }
        }

    }

}
