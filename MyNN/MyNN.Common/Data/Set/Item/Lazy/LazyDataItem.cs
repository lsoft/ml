using System;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Other;

namespace MyNN.Common.Data.Set.Item.Lazy
{
    [Serializable]
    public class LazyDataItem : IDataItem
    {
        private readonly float[] _input;
        private readonly float[] _output;
        private readonly INoiser _noiser;
        private readonly ISerializationHelper _serializationHelper;

        public int InputLength
        {
            get
            {
                return
                    this._input.Length;
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

        public float[] Input
        {
            get
            {
                //клонируем массив, чтобы не испортить оригинал
                var clonedInput = _input.CloneArray();

                //клонировать надо, чтобы каждый запрос LazyDataItem.Input выдавал одно и то же
                var clonedNoiser = _serializationHelper.DeepClone(_noiser);

                //применяем шум
                var result = clonedNoiser.ApplyNoise(clonedInput);

                return result;
            }
        }

        public float[] Output
        {
            get
            {
                return
                    _output;
            }
        }

        public LazyDataItem(
            float[] input,
            float[] output,
            INoiser noiser,
            ISerializationHelper serializationHelper
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
            if (noiser == null)
            {
                throw new ArgumentNullException("noiser");
            }
            if (serializationHelper == null)
            {
                throw new ArgumentNullException("serializationHelper");
            }

            _input = input;
            _output = output;
            _noiser = noiser;
            _serializationHelper = serializationHelper;
        }
    }
}