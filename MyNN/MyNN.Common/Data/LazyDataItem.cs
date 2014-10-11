using System;
using MyNN.Common.Data.TrainDataProvider.Noiser;
using MyNN.Common.Other;

namespace MyNN.Common.Data
{
    [Serializable]
    public class LazyDataItem : IDataItem
    {
        private readonly IDataItem _dataItem;
        private readonly INoiser _noiser;
        private readonly ISerializationHelper _serializationHelper;

        public int InputLength
        {
            get
            {
                return
                    _dataItem.InputLength;
            }
        }

        public int OutputLength
        {
            get
            {
                return
                    _dataItem.OutputLength;
            }
        }

        public int OutputIndex
        {
            get
            {
                return
                    _dataItem.OutputIndex;
            }
        }

        public float[] Input
        {
            get
            {
                //Запрашиваем массив из внутреннего итема
                var inner = _dataItem.Input;

                //клонировать надо, чтобы каждый запрос LazyDataItem.Input выдавал одно и то же
                var clonedNoiser = _serializationHelper.DeepClone(_noiser);

                //применяем шум
                var result = clonedNoiser.ApplyNoise(inner);

                return result;
            }
        }

        public float[] Output
        {
            get
            {
                return
                    _dataItem.Output;
            }
        }

        public LazyDataItem(
            IDataItem dataItem,
            INoiser noiser,
            ISerializationHelper serializationHelper
            )
        {
            if (dataItem == null)
            {
                throw new ArgumentNullException("dataItem");
            }
            if (noiser == null)
            {
                throw new ArgumentNullException("noiser");
            }
            if (serializationHelper == null)
            {
                throw new ArgumentNullException("serializationHelper");
            }

            _dataItem = dataItem;
            _noiser = noiser;
            _serializationHelper = serializationHelper;
        }
    }
}