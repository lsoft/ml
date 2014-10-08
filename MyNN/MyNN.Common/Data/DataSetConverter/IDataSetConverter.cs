namespace MyNN.Common.Data.DataSetConverter
{
    public interface IDataSetConverter
    {
        IDataSet Convert(IDataSet beforeTransformation);
    }
}
