using System;

namespace MyNN.Common.NewData.DataSet.ItemTransformation
{
    public class DataItemTransformationFactory : IDataItemTransformationFactory
    {
        private readonly Func<int, IDataItemTransformation> _dataItemTransformationFunc;

        public bool IsAutoencoderDataSet
        {
            get
            {
                return
                    _dataItemTransformationFunc(0).IsAutoencoderDataSet;
            }
        }

        public DataItemTransformationFactory(
            Func<int, IDataItemTransformation> dataItemTransformationFunc
            )
        {
            if (dataItemTransformationFunc == null)
            {
                throw new ArgumentNullException("dataItemTransformationFunc");
            }

            _dataItemTransformationFunc = dataItemTransformationFunc;
        }


        public IDataItemTransformation CreateTransformation(int epochNumber)
        {
            return
                _dataItemTransformationFunc(epochNumber);
        }
    }
}