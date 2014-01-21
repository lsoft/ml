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

        public static T[] Concatenate<T>(this T[] a, T[] b)
        {
            if (a == null)
            {
                throw new ArgumentNullException("a");
            }
            if (b == null)
            {
                throw new ArgumentNullException("b");
            }

            var r = new T[a.Length + b.Length];
            Array.Copy(a, 0, r, 0, a.Length);
            Array.Copy(b, 0, r, a.Length, b.Length);

            return
                r;
        }

        public static T[] GetSubArray<T>(this T[] a, int startIndex)
        {
            if (a == null)
            {
                throw new ArgumentNullException("a");
            }
            if (startIndex < 0 || startIndex >= a.Length)
            {
                throw new ArgumentException("startIndex < 0 || startIndex >= a.Length");
            }

            return
                GetSubArray(a, startIndex, a.Length - startIndex);
        }


        public static T[] GetSubArray<T>(this T[] a, int startIndex, int length)
        {
            if (a == null)
            {
                throw new ArgumentNullException("a");
            }
            if (startIndex < 0 || startIndex >= a.Length)
            {
                throw new ArgumentException("startIndex < 0 || startIndex >= a.Length");
            }
            if (length < 0 || (startIndex + length) > a.Length)
            {
                throw new ArgumentException("length < 0 || (startIndex + length) >= a.Length");
            }

            var r = new T[length];
            Array.Copy(a, startIndex, r, 0, length);


            return r;
        }
    }

}
