using System;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;

namespace MyNN.Common.NewData.DataSet.Iterator
{
    public interface IDataIteratorFactory
    {
        IDataIterator CreateIterator(
            IDataItemLoader itemLoader,
            Func<IDataItemTransformation> transformationFactory
            );
    }
}