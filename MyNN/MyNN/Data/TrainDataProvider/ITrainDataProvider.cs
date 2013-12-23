namespace MyNN.Data.TrainDataProvider
{
    public interface ITrainDataProvider
    {
        bool IsAuencoderDataSet
        {
            get;
        }

        DataSet GetDeformationDataSet(int epocheNumber);
    }
}
