using System.Collections.Generic;
using MyNN.Data;

namespace MyNN
{
    public interface ISerializationHelper
    {
        List<DataItem> ReadDataFromFile(string fileName, int totalCount);

        void SaveDataToFile(List<DataItem> obj, string fileName);

        T LoadLastFile<T>(string dirname, string mask)
            where T : class;

        T LoadFromFile<T>(string fileName)
            where T : class;

        void SaveToFile<T>(T obj, string fileName);

        T DeepClone<T>(T obj);
    }
}