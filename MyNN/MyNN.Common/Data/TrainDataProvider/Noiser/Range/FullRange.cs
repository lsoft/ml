using System;
using MyNN.Common.Other;

namespace MyNN.Common.Data.TrainDataProvider.Noiser.Range
{
    [Serializable]
    public class FullRange : IRange
    {

        public FullRange(
            )
        {
        }

        public bool[] GetIndexMask(int dataLength)
        {
            var result = new bool[dataLength];
            result.Fill(true);

            return result;
        }
    }
}