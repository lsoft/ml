﻿using System;

namespace MyNN.Common.Other
{
    public class KahanAlgorithm
    {
        public class Accumulator
        {
            public float Sum;
            public float C;

            public Accumulator(
                )
            {
                this.Sum = 0f;
                this.C = 0f;
            }
        }

        public static void AddElement(
            ref Accumulator acc,
            float dataItem
            )
        {
            var y = dataItem - acc.C;
            var t = acc.Sum + y;
            acc.C = (t - acc.Sum) - y;
            acc.Sum = t;
        }

        
        public static float Sum(
            int dataCount,
            Func<int, float> floatProvider
            )
        {
            if (floatProvider == null)
            {
                throw new ArgumentNullException("floatProvider");
            }

            if (dataCount == 0)
            {
                return 0f;
            }

            var tempArray = new float[dataCount];
            for (var index = 0; index < dataCount; index++)
            {
                tempArray[index] = floatProvider(index);
            }

            return
                KahanAlgorithm.Sum(tempArray);
        }

        public static float Sum(
            float[] data)
        {
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            if (data.Length == 0)
            {
                return 0f;
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
