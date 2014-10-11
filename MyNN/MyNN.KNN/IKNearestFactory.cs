using MyNN.Common.Data;
using MyNN.Common.Data.Set;

namespace MyNN.KNN
{
    public interface IKNearestFactory
    {
        IKNearest CreateKNearest(IDataSet dataList);
    }
}