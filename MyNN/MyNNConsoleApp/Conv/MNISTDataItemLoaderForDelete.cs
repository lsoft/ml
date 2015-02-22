using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.OutputConsole;

namespace MyNNConsoleApp.Conv
{
    [Serializable]
    public class MNISTDataItemLoaderForDelete : IDataItemLoader
    {
        private readonly INormalizer _normalizer;
        private readonly List<IDataItem> _list;

        public int Count
        {
            get
            {
                return
                    _list.Count;
            }
        }

        public MNISTDataItemLoaderForDelete(
            string root,
            int maxCount,
            bool binarize,
            IDataItemFactory dataItemFactory,
            INormalizer normalizer
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
            if (normalizer == null)
            {
                throw new ArgumentNullException("normalizer");
            }

            _normalizer = normalizer;
            _list = GetDataList(
                root,
                maxCount,
                binarize,
                dataItemFactory
                );
        }

        public IDataItem Load(int index)
        {
            return
                _list[index];
        }

        public void Normalize(float bias = 0f)
        {
            foreach (var di in this._list)
            {
                _normalizer.Normalize(di.Input, bias);
            }
        }

        public void GNormalize()
        {
            foreach (var di in this._list)
            {
                _normalizer.GNormalize(di.Input);
            }
        }


        private static List<IDataItem> GetDataList(
            string root,
            int maxCount,
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

            ConsoleAmbientContext.Console.Write("Processing images...  ");
            var till = DateTime.Now;

            var resultList = new List<IDataItem>();

            //������� ���� � �������
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

                //������� ���� � �������
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

                    //������ ��������
                    var imageBuffer = new byte[imageHeight * imageWidth * imageCount];
                    trainSet.Read(imageBuffer, 0, imageHeight * imageWidth * imageCount);

                    for (var imageIndex = 0; imageIndex < Math.Min((long)imageCount, maxCount); imageIndex++)
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
                        doutput[labelsb[imageIndex]] = 1f;

                        var d = dataItemFactory.CreateDataItem(
                            dinput,
                            doutput);

                        resultList.Add(d);
                    }
                }
            }

            ConsoleAmbientContext.Console.WriteLine("takes " + (DateTime.Now - till));
            ConsoleAmbientContext.Console.WriteLine(
                "Loaded {0} items",
                resultList.Count
                );

            return
                resultList;
        }

    }
}