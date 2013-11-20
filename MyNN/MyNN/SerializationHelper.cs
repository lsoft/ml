using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using MyNN.Data;
using MyNN.NeuralNet.Train;

namespace MyNN
{
    public static class SerializationHelper
    {
        public static List<DataItem> ReadDataFromFile(string fileName, int totalCount)
        {
            using (var reader = new BinaryReader(File.Open(fileName, FileMode.Open)))
            {
                var dataCount = Math.Min(totalCount, reader.ReadInt32());

                Console.WriteLine("Loading " + fileName + ", total: " + dataCount);

                var result = new List<DataItem>(dataCount + 1);//1 - запас

                for (var cc = 0; cc < dataCount; cc++)
                {
                    var iCount = reader.ReadInt32();

                    var i = new float[iCount];
                    for (var ii = 0; ii < iCount; ii++)
                    {
                        i[ii] = reader.ReadSingle();
                    }

                    var oCount = reader.ReadInt32();
                    var o = new float[oCount];
                    for (var oo = 0; oo < oCount; oo++)
                    {
                        o[oo] = reader.ReadSingle();
                    }

                    var di = new DataItem(i, o);
                    result.Add(di);
                }

                return result;
            }
        }

        public static void SaveDataToFile(List<DataItem> obj, string fileName)
        {
            
            using (var writer = new BinaryWriter(File.Open(fileName, FileMode.Create)))
            {
                writer.Write(obj.Count);

                foreach(var o in obj)
                {
                    writer.Write(o.Input.Length);

                    foreach (var id in o.Input)
                    {
                        writer.Write(id);
                    }

                    writer.Write(o.Output.Length);
                    foreach (var od in o.Output)
                    {
                        writer.Write(od);
                    }
                }
            }
        }

        public static T LoadLastFile<T>(string dirname, string mask)
            where T : class
        {
            var files = Directory.GetFiles(dirname, mask);

            if (files == null || files.Length == 0)
            {
                return null;
            }

            var lastFile =
                (from x in files
                orderby File.GetLastWriteTime(x) descending 
                select x).First();

            return
                LoadFromFile<T>(lastFile);
        }

        public static T LoadFromFile<T>(string fileName)
            where T : class
        {
            Console.WriteLine("Loading " + fileName);

            var formatter = new BinaryFormatter();
            using (Stream stream = File.Open(fileName, FileMode.Open))
            {
                return (T)formatter.Deserialize(stream);
            }
        }

        public static void SaveToFile<T>(T obj, string fileName)
        {
            var formatter = new BinaryFormatter();
            using (Stream stream = File.Open(fileName, FileMode.Create))
            {
                formatter.Serialize(stream, obj);
            }
        }

        public static T DeepClone<T>(T obj)
        {
            T result;

            var formatter = new BinaryFormatter();
            using (Stream stream = new MemoryStream(1000000))
            {
                formatter.Serialize(stream, obj);

                stream.Position = 0;

                result = (T)formatter.Deserialize(stream);
            }

            return result;
        }
    }
}
