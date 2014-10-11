using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Other;

namespace MyNN.Common.Data.Set.Item.Lazy
{
    [Serializable]
    public class LazyDataItem : IDataItem
    {
        private readonly Pair<int, float>[] _inputTable;
        private readonly Pair<int, float>[] _outputTable;

        private readonly INoiser _noiser;
        private readonly ISerializationHelper _serializationHelper;

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

        public float[] Input
        {
            get
            {
                //клонируем массив, чтобы не испортить оригинал
                var clonedInput = new float[InputLength];

                foreach (var ii in _inputTable)
                {
                    clonedInput[ii.First] = ii.Second;
                }

                //клонировать нойзер надо, чтобы каждый запрос LazyDataItem.Input выдавал одно и то же
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
                //клонируем массив, чтобы не испортить оригинал
                var result = new float[OutputLength];

                foreach (var ii in _outputTable)
                {
                    result[ii.First] = ii.Second;
                }

                return result;
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

            _noiser = noiser;
            _serializationHelper = serializationHelper;

            InputLength = input.Length;
            OutputLength = output.Length;

            _inputTable = ConvertToTable(input).ToArray();
            _outputTable = ConvertToTable(output).ToArray();
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