namespace MyNN.Common.Data.Set.Item
{
    public interface IDataItemFactory
    {
        IDataItem CreateDataItem(
            float[] input,
            float[] output
            );
    }
}