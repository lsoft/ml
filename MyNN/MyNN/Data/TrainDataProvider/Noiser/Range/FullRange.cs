﻿namespace MyNN.Data.TrainDataProvider.Noiser.Range
{
    public class FullRange : IRange
    {
        private readonly int _dataLength;

        public FullRange(
            int dataLength)
        {
            _dataLength = dataLength;
        }

        public bool[] GetIndexMask()
        {
            var result = new bool[_dataLength];
            result.Fill(true);

            return result;
        }
    }
}