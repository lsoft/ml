using MyNN.Common.Data.Set.Item;

namespace MyNN.Common.NewData.DataSet.ItemLoader
{
    public interface IDataItemLoader : IDataItemLoaderProperties
    {
        IDataItem Load(
            int index
            );
    }
}