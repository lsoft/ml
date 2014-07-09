using MyNN.Data;

namespace MyNN.KNN
{
    public interface IKNearestFactory
    {
        IKNearest CreateKNearest(DataSet dataList);
    }
}