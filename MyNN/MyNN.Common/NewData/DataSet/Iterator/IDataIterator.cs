using System.Collections.Generic;
using MyNN.Common.NewData.Item;

namespace MyNN.Common.NewData.DataSet.Iterator
{
    public interface IDataIterator : IEnumerator<IDataItem>
    {
        int Count
        {
            get;
        }

    }
}