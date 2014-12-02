using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;

namespace MyNN.Common.Data.DataLoader
{
    public class MNISTDataLoader
    {
        public static IDataSet GetDataSet(
            string root,
            int maxCountFilesInCategory,
            bool binarize,
            IDataItemFactory dataItemFactory
            )
        {
            if (root == null)
            {
                throw new ArgumentNullException("root");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }

            Console.WriteLine("Processing images...");
            var till = DateTime.Now;

            var resultList = new List<IDataItem>();

            //готовим файл с данными
            using (var trainSet = File.OpenRead(root + "\\images.idx3-ubyte"))
            {
                {
                    var magicNumb = new byte[4];
                    trainSet.Read(magicNumb, 0, 4);

                    var magicNum = BitConverter.ToInt32(magicNumb, 0);
                    if (magicNum != 0x03080000)
                    {
                        throw new Exception("cannot find magic number");
                    }
                }

                var imageCountb = new byte[4];
                trainSet.Read(imageCountb, 0, 4);

                var imageHeightb = new byte[4];
                trainSet.Read(imageHeightb, 0, 4);

                var imageWidthb = new byte[4];
                trainSet.Read(imageWidthb, 0, 4);

                var imageCount = BitConverter.ToInt32(imageCountb.Reverse().ToArray(), 0);
                var imageHeight = BitConverter.ToInt32(imageHeightb.Reverse().ToArray(), 0);
                var imageWidth = BitConverter.ToInt32(imageWidthb.Reverse().ToArray(), 0);

                //готовим файл с метками
                using (var trainLabelSet = File.OpenRead(root + "\\labels.idx1-ubyte"))
                {
                    {
                        var magicNumb = new byte[4];
                        trainLabelSet.Read(magicNumb, 0, 4);

                        var magicNum = BitConverter.ToInt32(magicNumb, 0);
                        if (magicNum != 0x01080000)
                        {
                            throw new Exception("cannot find magic number");
                        }
                    }

                    var labelCountb = new byte[4];
                    trainLabelSet.Read(labelCountb, 0, 4);

                    var labelCount = BitConverter.ToInt32(labelCountb.Reverse().ToArray(), 0);

                    var labelsb = new byte[labelCount];
                    trainLabelSet.Read(labelsb, 0, labelCount);

                    //читаем картинку
                    var imageBuffer = new byte[imageHeight * imageWidth * imageCount];
                    trainSet.Read(imageBuffer, 0, imageHeight * imageWidth * imageCount);

                    for (var imageIndex = 0; imageIndex < Math.Min((long)imageCount, (long)(maxCountFilesInCategory) * 10); imageIndex++)
                    {
                        var dinput = new float[784];

                        var inImageIndex = 0;
                        for (var h = 0; h < imageHeight; h++)
                        {
                            for (var w = 0; w < imageWidth; w++)
                            {
                                var value = imageBuffer[(imageIndex * imageHeight * imageWidth) + inImageIndex];

                                dinput[inImageIndex] = 
                                    binarize
                                        ? (value >= 128 ? 1f : 0f)
                                        : value / 255.0f;

                                inImageIndex++;
                            }
                        }

                        var doutput = new float[10];
                        var outputIndex = labelsb[imageIndex];
                        for (var cc = 0; cc < 10; cc++)
                        {
                            doutput[cc] = cc == outputIndex ? 1.0f : 0.0f;
                        }

                        var d = dataItemFactory.CreateDataItem(
                            dinput,
                            doutput);

                        resultList.Add(d);
                    }
                }
            }

            Console.WriteLine("processing images takes " + (int)(DateTime.Now - till).TotalSeconds + " sec");

            return
                new DataSet(
                    resultList);
        }
    }
}
