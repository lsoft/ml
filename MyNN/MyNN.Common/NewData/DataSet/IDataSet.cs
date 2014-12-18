using System.Collections.Generic;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.Item;

namespace MyNN.Common.NewData.DataSet
{
    public interface IDataSet : IEnumerable<IDataItem>
    {
        int Count
        {
            get;
        }

        bool IsAutoencoderDataSet
        {
            get;
        }

        int InputLength
        {
            get;
        }

        int OutputLength
        {
            get;
        }

        IDataIterator StartIterate(
            );
    }
}