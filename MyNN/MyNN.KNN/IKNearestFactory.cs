using System.Collections.Generic;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;

namespace MyNN.KNN
{
    public interface IKNearestFactory
    {
        IKNearest CreateKNearest(IList<IDataItem> dataList);
    }
}