using System;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;

namespace MyNN.Common.NewData.DataSet.Iterator
{
    public class CacheDataIteratorFactory : IDataIteratorFactory
    {
        private readonly int _cacheCount;
        private readonly IDataIteratorFactory _dataIteratorFactory;

        public CacheDataIteratorFactory(
            int cacheCount,
            IDataIteratorFactory dataIteratorFactory
            )
        {
            if (dataIteratorFactory == null)
            {
                throw new ArgumentNullException("dataIteratorFactory");
            }

            _cacheCount = cacheCount;
            _dataIteratorFactory = dataIteratorFactory;
        }

        public IDataIterator CreateIterator(
            IDataItemLoader itemLoader,
            Func<IDataItemTransformation> transformationFactory
            )
        {
            if (itemLoader == null)
            {
                throw new ArgumentNullException("itemLoader");
            }
            if (transformationFactory == null)
            {
                throw new ArgumentNullException("transformationFactory");
            }

            var iterator = _dataIteratorFactory.CreateIterator(
                itemLoader,
                transformationFactory
                );

            return
                new CacheDataIterator(
                    _cacheCount,
                    iterator
                    );
        }
    }
}