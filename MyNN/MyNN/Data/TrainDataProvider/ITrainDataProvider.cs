namespace MyNN.Data.TrainDataProvider
{
    public interface ITrainDataProvider
    {
        bool IsAuencoderDataSet
        {
            get;
        }

        bool IsClassificationAuencoderDataSet
        {
            get;
        }

        DataSet GetDataSet(int epocheNumber);
    }
}
