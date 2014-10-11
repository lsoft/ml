using System;

namespace MyNN.Common.Data
{
    [Serializable]
    public class SparseDataItem : IDataItem
    {
        private readonly int[] _inputIndex;
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
            int[] inputIndex,
            float[] output
            )
        {
            if (inputIndex == null)
            {
                throw new ArgumentNullException("inputIndex");
            }
            if (output == null)
            {
                throw new ArgumentNullException("output");
            }

            InputLength = inputLength;
            _inputIndex = inputIndex;
            _output = output;
        }

        public float[] Input
        {
            get
            {
                var result = new float[InputLength];

                foreach (var ii in _inputIndex)
                {
                    result[ii] = 1f;
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