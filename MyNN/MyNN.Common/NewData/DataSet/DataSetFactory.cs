using System;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;

namespace MyNN.Common.NewData.DataSet
{
    public class DataSetFactory : IDataSetFactory
    {
        private readonly IDataIteratorFactory _iteratorFactory;
        private readonly IDataItemTransformationFactory _itemTransformationFactory;

        public DataSetFactory(
            IDataIteratorFactory iteratorFactory,
            IDataItemTransformationFactory itemTransformationFactory
            )
        {
            if (iteratorFactory == null)
            {
                throw new ArgumentNullException("iteratorFactory");
            }
            if (itemTransformationFactory == null)
            {
                throw new ArgumentNullException("itemTransformationFactory");
            }

            _iteratorFactory = iteratorFactory;
            _itemTransformationFactory = itemTransformationFactory;
        }

        public IDataSet CreateDataSet(
            IDataItemLoader itemLoader, 
            int epochNumber
            )
        {
            return
                new DataSet(
                    _iteratorFactory,
                    _itemTransformationFactory,
                    itemLoader,
                    epochNumber
                    );
        }
    }
}