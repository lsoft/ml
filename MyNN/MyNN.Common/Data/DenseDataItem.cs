using System;

namespace MyNN.Common.Data
{
    [Serializable]
    public class DenseDataItem : IDataItem
    {
        private readonly float[] _input;
        private readonly float[] _output;

        public int InputLength
        {
            get
            {
                return
                    _input.Length;
            }
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

        private DenseDataItem()
        {
        }

        public DenseDataItem(
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

            _input = input;
            _output = output;
        }

        public float[] Input
        {
            get
            {
                return _input;
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
    //*/
}
