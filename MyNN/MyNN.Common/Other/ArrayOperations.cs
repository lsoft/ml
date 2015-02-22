using System;

namespace MyNN.Common.Other
{
    public static class ArrayOperations
    {
        public static T[] CloneAndAppend<T>(this T[] a, T b)
        {
            var r = new T[a.Length + 1];
            a.CopyTo(r, 0);
            r[a.Length] = b;

            return r;
        }

        public static void Clear<T>(this T[] a, int startIndex, int length)
        {
            for (var cc = startIndex; cc < startIndex + length; cc++)
            {
                a[cc] = default (T);
            }
        }

        public static void Clear<T>(this T[] a)
        {
            for (var cc = 0; cc < a.Length; cc++)
            {
                a[cc] = default(T);
            }
        }

        public static T[] CloneArray<T>(this T[] a)
        {
            if (a == null)
            {
                return null;
            }

            var result = new T[a.Length];

            for (var cc = 0; cc < a.Length; cc++)
            {
                result[cc] = a[cc];
            }

            return result;
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

        public static void Fill<T>(this T[] a, Func<int, T> value)
        {
            for (var cc = 0; cc < a.Length; cc++)
            {
                a[cc] = value(cc);
            }
        }

        public static void Transform<T>(this T[] a, Func<int, T, T> value)
        {
            for (var cc = 0; cc < a.Length; cc++)
            {
                a[cc] = value(cc, a[cc]);
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

        public static T[] RemoveLastElement<T>(this T[] a)
        {
            if (a == null)
            {
                throw new ArgumentNullException("a");
            }

            return
                GetSubArray(a, 0, a.Length - 1);
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

        public static bool ValuesAreEqual(int[] array0, int[] array1)
        {
            if (array0 == null && array1 == null)
            {
                return true;
            }
            if (array0 != null && array1 == null)
            {
                return false;
            }
            if (array0 == null && array1 != null)
            {
                return false;
            }
            if (array0.Length != array1.Length)
            {
                return false;
            }

            for (var index = array0.Length; index < array0.Length; index++)
            {
                if (array0[index] != array1[index])
                {
                    return false;
                }
            }

            return true;
        }

        public static bool ValuesAreEqual(float[] array0, float[] array1, float epsilon, out float maxDiff, out int maxDiffIndex)
        {
            if (array0 == null && array1 == null)
            {
                maxDiff = 0;
                maxDiffIndex = 0;
                return true;
            }
            if (array0 != null && array1 == null)
            {
                maxDiff = float.MaxValue;
                maxDiffIndex = 0;
                return false;
            }
            if (array0 == null && array1 != null)
            {
                maxDiff = float.MaxValue;
                maxDiffIndex = 0;
                return false;
            }
            if (array0.Length != array1.Length)
            {
                maxDiff = float.MaxValue;
                maxDiffIndex = 0;
                return false;
            }

            maxDiff = 0;
            maxDiffIndex = 0;
            for (var index = 0; index < array0.Length; index++)
            {
                var currentDiff = (array0[index] >= array1[index] ? array0[index] - array1[index] : array1[index] - array0[index]);

                if (currentDiff > maxDiff)
                {
                    maxDiff = currentDiff;
                    maxDiffIndex = index;
                }
            }

            if (maxDiff > epsilon)
            {
                return false;
            }

            return true;
        }
        public static bool ValuesAreEqual(float[] array0, float[] array1, float epsilon, out float maxDiff)
        {
            if (array0 == null && array1 == null)
            {
                maxDiff = 0;
                return true;
            }
            if (array0 != null && array1 == null)
            {
                maxDiff = float.MaxValue;
                return false;
            }
            if (array0 == null && array1 != null)
            {
                maxDiff = float.MaxValue;
                return false;
            }
            if (array0.Length != array1.Length)
            {
                maxDiff = float.MaxValue;
                return false;
            }

            maxDiff = 0;
            for (var index = 0; index < array0.Length; index++)
            {
                var currentDiff = (array0[index] >= array1[index] ? array0[index] - array1[index] : array1[index] - array0[index]);

                if (currentDiff > maxDiff)
                {
                    maxDiff = currentDiff;
                }
            }

            if (maxDiff > epsilon)
            {
                return false;
            }

            return true;
        }

        public static TT[] ConvertAll<TF, TT>(this TF[] array, Func<TF, TT> converter)
        {
            if (array == null)
            {
                throw new ArgumentNullException("array");
            }
            if (converter == null)
            {
                throw new ArgumentNullException("converter");
            }

            var result = new TT[array.Length];

            for (var cc = 0; cc < array.Length; cc++)
            {
                result[cc] = converter(array[cc]);
            }

            return result;
        }

        public static void MapInto(
            this float[] a0,
            float[] a1,
            Func<float, float, float> mapper)
        {
            if (a0 == null)
            {
                throw new ArgumentNullException("a0");
            }
            if (a1 == null)
            {
                throw new ArgumentNullException("a1");
            }
            if (mapper == null)
            {
                throw new ArgumentNullException("mapper");
            }
            if (a0.Length != a1.Length)
            {
                throw new ArgumentException("a0.Length != a1.Length");
            }

            for (var cc = 0; cc < a0.Length; cc++)
            {
                a0[cc] = mapper(a0[cc], a1[cc]);
            }
        }

        public static float[] Map(
            float[] a0, 
            float[] a1, 
            Func<float, float, float> mapper)
        {
            if (a0 == null)
            {
                throw new ArgumentNullException("a0");
            }
            if (a1 == null)
            {
                throw new ArgumentNullException("a1");
            }
            if (mapper == null)
            {
                throw new ArgumentNullException("mapper");
            }
            if (a0.Length != a1.Length)
            {
                throw new ArgumentException("a0.Length != a1.Length");
            }

            var result = new float[a0.Length];

            for (var cc = 0; cc < a0.Length; cc++)
            {
                result[cc] = mapper(a0[cc], a1[cc]);
            }

            return result;
        }

        public static int Sum(
            this int[] array)
        {
            if (array == null)
            {
                throw new ArgumentNullException("array");
            }

            var result = 0;

            for (var cc = 0; cc < array.Length; cc++)
            {
                result += array[cc];
            }

            return result;
        }

        public static float Sum(
            this float[] array)
        {
            if (array == null)
            {
                throw new ArgumentNullException("array");
            }

            var result = 0f;

            for (var cc = 0; cc < array.Length; cc++)
            {
                result += array[cc];
            }

            return result;
        }

        public static float[] DiffArrays(
            float[] first,
            float[] second
            )
        {
            if (first == null)
            {
                throw new ArgumentNullException("first");
            }
            if (second == null)
            {
                throw new ArgumentNullException("second");
            }
            if (first.Length != second.Length)
            {
                throw new ArgumentException("first.Length != second.Length");
            }

            var result = new float[first.Length];

            for (var cc = 0; cc < first.Length; cc++)
            {
                result[cc] = first[cc] - second[cc];
            }

            return result;
        }
    }

}
