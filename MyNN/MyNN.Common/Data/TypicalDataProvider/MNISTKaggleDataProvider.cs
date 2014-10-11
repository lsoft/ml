using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.Set.Item.Dense;

namespace MyNN.Common.Data.TypicalDataProvider
{
    public class MNISTKaggleDataProvider
    {
        public static IDataSet GetDataSet(
            string root,
            string filename,
            bool isTrainSet,
            int maxCountFilesInCategory,
            bool binarize,
            IDataItemFactory dataItemFactory
            )
        {
            if (root == null)
            {
                throw new ArgumentNullException("root");
            }
            if (filename == null)
            {
                throw new ArgumentNullException("filename");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            Console.Write("Processing images...");
            var till = DateTime.Now;

            var resultList = new List<IDataItem>();

            var c0 = File.ReadAllLines(Path.Combine(root, filename));
            var c1 = c0.Skip(1);
            var iterCount = 0;
            foreach (var c in c1)
            {
                var l = c.Split(',');

                if (isTrainSet)
                {

                    var label = int.Parse(l[0]);
                    var pixels = l.Skip(1).ToList().ConvertAll(j => float.Parse(j) / 255.0f);

                    //var bmp = new Bitmap(28, 28);
                    //CreateBitmap(bmp, 28, 28, 0, pixels.ToArray());

                    //bmp.Save(label + ".bmp");

                    //Console.WriteLine(label);
                    //Console.WriteLine(pixels);

                    var dinput = pixels.ConvertAll(j => (float)j).ToArray();
                    var doutput = new float[10];

                    for (var cc = 0; cc < 10; cc++)
                    {
                        doutput[cc] = cc == label ? 1.0f : 0.0f;
                    }

                    var d = dataItemFactory.CreateDataItem(
                        dinput,
                        doutput
                        );

                    resultList.Add(d);
                }
                else
                {
                    var pixels = l.ToList().ConvertAll(j => float.Parse(j) / 255.0f);

                    //var bmp = new Bitmap(28, 28);
                    //CreateBitmap(bmp, 28, 28, 0, pixels.ToArray());

                    //bmp.Save(label + ".bmp");

                    //Console.WriteLine(label);
                    //Console.WriteLine(pixels);

                    var input = pixels.ConvertAll(
                        j =>
                            binarize
                            ? (j >= 0.5f ? 1f : 0f)
                            : j).ToArray();
                    var output = new float[10];

                    var d = dataItemFactory.CreateDataItem(
                        input,
                        output);

                    resultList.Add(d);
                }

                if (++iterCount >= maxCountFilesInCategory)
                {
                    break;
                }
            }

            Console.WriteLine(" takes " + (int)(DateTime.Now - till).TotalSeconds + " sec, loaded " + resultList.Count + " records");

            return
                new DataSet(
                    resultList);
        }
    }
}
