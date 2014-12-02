using System;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;

namespace MyNN.Common.NewData.DataSet.Iterator
{
    public class DataIteratorFactory : IDataIteratorFactory
    {
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

            return 
                new DataIterator(
                    itemLoader,
                    transformationFactory
                    );
        }
    }
}