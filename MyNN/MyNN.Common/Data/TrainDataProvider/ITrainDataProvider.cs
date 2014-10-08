namespace MyNN.Common.Data.TrainDataProvider
{
    public interface ITrainDataProvider
    {
        IDataSet GetDataSet(int epocheNumber);
    }
}
