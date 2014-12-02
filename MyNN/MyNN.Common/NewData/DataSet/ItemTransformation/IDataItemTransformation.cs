using MyNN.Common.Data.Set.Item;

namespace MyNN.Common.NewData.DataSet.ItemTransformation
{
    public interface IDataItemTransformation : IDataTransformProperties
    {
        IDataItem Transform(
            IDataItem before
            );
    }
}