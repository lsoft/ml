using MyNN.Common.NewData.DataSet.ItemLoader;

namespace MyNN.Common.NewData.DataSet
{
    public interface IDataSetFactory
    {
        IDataSet CreateDataSet(
            IDataItemLoader itemLoader,
            int epochNumber
            );
    }
}