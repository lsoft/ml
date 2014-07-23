namespace MyNN.Data.TrainDataProvider
{
    public interface ITrainDataProvider
    {
        bool IsAuencoderDataSet
        {
            get;
        }

        IDataSet GetDataSet(int epocheNumber);
    }
}
