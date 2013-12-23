using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.MLP2.Randomizer;

namespace MyNN.Data.TypicalDataProvider
{
    public class MNISTDataProvider
    {
        public static DataSet GetDataSet(
            string root,
            int maxCountFilesInCategory,
            bool binarize = false)
        {
            Console.WriteLine("Processing images...");
            var till = DateTime.Now;

            var resultList = new List<DataItem>();

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
                        var d = new DataItem();
                        d.Input = new float[784];
                        d.Output = new float[10];

                        var inImageIndex = 0;
                        for (var h = 0; h < imageHeight; h++)
                        {
                            for (var w = 0; w < imageWidth; w++)
                            {
                                var value = imageBuffer[(imageIndex * imageHeight * imageWidth) + inImageIndex];

                                d.Input[inImageIndex] = 
                                    binarize
                                        ? (value >= 128 ? 1f : 0f)
                                        : value / 255.0f;

                                inImageIndex++;
                            }
                        }

                        var output = labelsb[imageIndex];
                        for (var cc = 0; cc < 10; cc++)
                        {
                            d.Output[cc] = cc == output ? 1.0f : 0.0f;
                        }

                        resultList.Add(d);
                    }
                }
            }

            Console.WriteLine("processing images takes " + (int)(DateTime.Now - till).TotalSeconds + " sec");

            return
                new DataSet(
                    resultList,
                    new MNISTVisualizer());
        }
    }
}
