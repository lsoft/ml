using System;

namespace MyNN.Common.Data
{
    [Serializable]
    public class DataItem
    {
        private float[] _input = null;
        private float[] _output = null;

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

        public DataItem()
        {
        }

        public DataItem(float[] input, float[] output)
        {
            _input = input;
            _output = output;
        }

        public float[] Input
        {
            get { return _input; }
            set { _input = value; }
        }

        public float[] Output
        {
            get { return _output; }
            set { _output = value; }
        }
    }
    //*/
}
