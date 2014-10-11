using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Common.Data;

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

                var itemsize = br.ReadInt32();

                var iteration = 0;
                while (fs.Position < fs.Length && iteration < maxValue)
                {
                    var input = new List<float>();
                    for (var cc = 0; cc < itemsize; cc++)
                    {
                        var onefloat = br.ReadSingle();
                        input.Add(onefloat);
                    }

                    var label = br.ReadInt32();

                    var output = new float[5];

                    if (label >= 0)
                    {
                        output[label] = 1f;
                    }

                    var di = new DenseDataItem(
                        input.ToArray(),
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
