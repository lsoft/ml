namespace MyNN.Common.NewData.DataSet.ItemTransformation
{
    public interface IDataItemTransformationFactory : IDataTransformProperties
    {
        IDataItemTransformation CreateTransformation(
            int epochNumber
            );
    }
}