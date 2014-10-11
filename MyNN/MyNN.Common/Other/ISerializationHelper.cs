using System.Collections.Generic;
using MyNN.Common.Data;

namespace MyNN.Common.Other
{
    public interface ISerializationHelper
    {
        List<IDataItem> ReadDataFromFile(string fileName, int totalCount);

        void SaveDataToFile(List<IDataItem> obj, string fileName);

        T LoadLastFile<T>(string dirname, string mask);

        T LoadFromFile<T>(string fileName);

        void SaveToFile<T>(T obj, string fileName);

        T DeepClone<T>(T obj);
    }
}