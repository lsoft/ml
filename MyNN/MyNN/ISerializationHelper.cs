namespace MyNN
{
    public interface ISerializationHelper
    {
        void SaveToFile<T>(T obj, string fileName);
        T DeepClone<T>(T obj);
    }
}