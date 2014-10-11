using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.Data;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.Set.Item.Sparse;
using MyNN.Common.Other;

namespace MyNNConsoleApp
{
    public class SAMRDataProvider
    {
        public static List<IDataItem> GetDataSet(
            string filepath, 
            int maxValue)
        {
            var result = new List<IDataItem>();

            using (var fs = new FileStream(filepath, FileMode.Open, FileAccess.Read))
            {
                var br = new BinaryReader(
                    fs);

                var itemSize = br.ReadInt32();

                var iteration = 0;
                while (fs.Position < fs.Length && iteration < maxValue)
                {
                    var indexCount = br.ReadInt32();

                    var inputIndex = new List<int>();
                    for (var cc = 0; cc < indexCount; cc++)
                    {
                        var oneIndex = br.ReadInt32();
                        inputIndex.Add(oneIndex);
                    }

                    var label = br.ReadInt32();

                    var output = new float[5];

                    if (label >= 0)
                    {
                        output[label] = 1f;
                    }

                    var di = new SparseDataItem(
                        itemSize,
                        inputIndex.ConvertAll(j => new Pair<int, float>(j, 1f)).ToArray(),
                        output
                        );

                    result.Add(di);

                    iteration++;
                }
            }

            return result;
        }
    }
}
