using System;
using MyNN.Common.NewData.Item;

namespace MyNN.Common.NewData.DataSet.ItemTransformation
{
    [Serializable]
    public class NoConvertDataItemTransformation : IDataItemTransformation
    {
        public bool IsAutoencoderDataSet
        {
            get
            {
                return
                    false;
            }
        }

        public NoConvertDataItemTransformation(
            )
        {
        }

        public IDataItem Transform(IDataItem before)
        {
            if (before == null)
            {
                throw new ArgumentNullException("before");
            }

            return
                before;
        }
    }
}