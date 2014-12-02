using MyNN.Common.NewData.DataSet;

namespace MyNN.Common.NewData.DataSetProvider
{
    public interface IDataSetProvider
    {
        int Count
        {
            get;
        }

        IDataSet GetDataSet(int epochNumber);

    }
}