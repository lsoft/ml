using System;
using MyNN.Common.NewData.Item;
using MyNN.Common.Other;

namespace MyNN.Common.NewData.DataSet.ItemTransformation
{

    [Serializable]
    public class ToAutoencoderDataItemTransformation : IDataItemTransformation
    {
        private readonly IDataItemFactory _dataItemFactory;

        public bool IsAutoencoderDataSet
        {
            get
            {
                return
                    true;
            }
        }

        public ToAutoencoderDataItemTransformation(
            IDataItemFactory dataItemFactory
            )
        {
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            _dataItemFactory = dataItemFactory;
        }

        public IDataItem Transform(IDataItem before)
        {
            if (before == null)
            {
                throw new ArgumentNullException("before");
            }

            var newItem = _dataItemFactory.CreateDataItem(before.Input, before.Input.CloneArray());

            return
                newItem;
        }
    }
}