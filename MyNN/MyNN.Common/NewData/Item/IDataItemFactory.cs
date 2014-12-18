namespace MyNN.Common.NewData.Item
{
    public interface IDataItemFactory
    {
        IDataItem CreateDataItem(
            float[] input,
            float[] output
            );
    }
}