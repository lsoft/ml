using MyNN.Common.Data;

namespace MyNN.KNN
{
    public interface IKNearestFactory
    {
        IKNearest CreateKNearest(IDataSet dataList);
    }
}