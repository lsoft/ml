using MyNN.Common.Data.Set;

namespace MyNN.Common.Data.TrainDataProvider
{
    public interface ITrainDataProvider
    {
        IDataSet GetDataSet(int epocheNumber);
    }
}
