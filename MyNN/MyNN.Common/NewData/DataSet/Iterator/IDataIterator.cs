using System.Collections.Generic;
using MyNN.Common.Data.Set.Item;

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