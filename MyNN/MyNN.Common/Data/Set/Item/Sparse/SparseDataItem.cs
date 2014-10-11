using System;
using System.Collections.Generic;
using MyNN.Common.Other;

namespace MyNN.Common.Data.Set.Item.Sparse
{
    [Serializable]
    public class SparseDataItem : IDataItem
    {
        private readonly Pair<int, float>[] _inputTable;
        private readonly float[] _output;

        public int InputLength
        {
            get;
            private set;
        }

        public int OutputLength
        {
            get
            {
                return
                    _output.Length;
            }
        }

        public int OutputIndex
        {
            get
            {
                var result = -1;

                for (var cc = 0; cc < _output.Length; cc++)
                {
                    if (_output[cc] >= float.Epsilon)
                    {
                        result = cc;
                        break;
                    }
                }

                return result;
            }
        }

        public SparseDataItem()
        {
        }

        public SparseDataItem(
            int inputLength,
            Pair<int, float>[] inputTable,
            float[] output
            )
        {
            if (inputTable == null)
            {
                throw new ArgumentNullException("inputTable");
            }
            if (output == null)
            {
                throw new ArgumentNullException("output");
            }

            InputLength = inputLength;
            _inputTable = inputTable;
            _output = output;
        }

        public SparseDataItem(
            float[] input,
            float[] output
            )
        {
            if (input == null)
            {
                throw new ArgumentNullException("input");
            }
            if (output == null)
            {
                throw new ArgumentNullException("output");
            }

            InputLength = input.Length;
            _output = output;

            var table = new List<Pair<int, float>>();
            for (var cc = 0; cc < input.Length; cc++)
            {
                var v = input[cc];

                if (v >= float.Epsilon || v <= -float.Epsilon)
                {
                    table.Add(
                        new Pair<int, float>(cc, v));
                }
            }
            _inputTable = table.ToArray();
        }

        public float[] Input
        {
            get
            {
                var result = new float[InputLength];

                foreach (var ii in _inputTable)
                {
                    result[ii.First] = ii.Second;
                }

                return result;
            }
        }

        public float[] Output
        {
            get
            {
                return _output;
            }
        }
    }
}