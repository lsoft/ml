using System.Collections.Generic;
using MyNN.Common.Data;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet;

namespace MyNN.KNN
{
    public interface IKNearestFactory
    {
        IKNearest CreateKNearest(IList<IDataItem> dataList);
    }
}