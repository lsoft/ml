using MyNN.Common.Data.Set;

namespace MyNN.Common.Data.DataSetConverter
{
    public interface IDataSetConverter
    {
        IDataSet Convert(IDataSet beforeTransformation);
    }
}
