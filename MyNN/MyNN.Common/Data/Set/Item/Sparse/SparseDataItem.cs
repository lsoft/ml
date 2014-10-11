using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Other;

namespace MyNN.Common.Data.Set.Item.Sparse
{
    [Serializable]
    public class SparseDataItem : IDataItem
    {
        private readonly Pair<int, float>[] _inputTable;
        private readonly Pair<int, float>[] _outputTable;

        public int InputLength
        {
            get;
            private set;
        }

        public int OutputLength
        {
            get;
            private set;
        }

        public int OutputIndex
        {
            get
            {
                var result = this._outputTable.Min(k => k.First);

                return result;
            }
        }

        public SparseDataItem()
        {
        }

        public SparseDataItem(
            int inputLength,
            int outputLength,
            Pair<int, float>[] inputTable,
            Pair<int, float>[] outputTable
            )
        {
            if (inputTable == null)
            {
                throw new ArgumentNullException("inputTable");
            }
            if (outputTable == null)
            {
                throw new ArgumentNullException("outputTable");
            }

            InputLength = inputLength;
            OutputLength = outputLength;
            _inputTable = inputTable;
            _outputTable = outputTable;
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
            OutputLength = output.Length;

            _inputTable = ConvertToTable(input).ToArray();
            _outputTable = ConvertToTable(output).ToArray();
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
                var result = new float[OutputLength];

                foreach (var ii in _outputTable)
                {
                    result[ii.First] = ii.Second;
                }

                return result;
            }
        }

        #region private method

        private List<Pair<int, float>> ConvertToTable(
            float[] array)
        {
            var table = new List<Pair<int, float>>();
            for (var cc = 0; cc < array.Length; cc++)
            {
                var v = array[cc];

                if (v >= float.Epsilon || v <= -float.Epsilon)
                {
                    table.Add(
                        new Pair<int, float>(cc, v));
                }
            }
            return table;
        }

        #endregion
    }
}