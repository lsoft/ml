using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Castle.DynamicProxy.Generators;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.LearningRateController;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSet.ItemTransformation;
using MyNN.Common.NewData.DataSet.Iterator;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.Mask.Factory;
using MyNN.MLP.Backpropagation;
using MyNN.MLP.Backpropagation.Metrics;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU;
using MyNN.MLP.DropConnect.Backpropagation.EpocheTrainer.DropConnect.OpenCL.GPU2;
using MyNN.MLP.DropConnect.Inferencer.Factory;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp
{
    public class CompareBPGPU
    {
        public static void DoCompare()
        {
            var orig = Do(false);
            var optimized = Do(true);

            Console.WriteLine(
                "============================ {0} ============================",
                (optimized - orig)
                );
        }

        private static float Do(
            bool useNew
            )
        {
            var randomizer = 
                //new DefaultRandomizer(1);
                new NoRandomRandomizer();

            var trainDataSetProvider = GetTrainProvider(
                10,
                false,
                false
                );
            
            var validationData = GetValidation(
                100,
                false,
                false
                );

            var serialization = new SerializationHelper();

            var mlpFactory = new MLPFactory(
                new LayerFactory(
                    new NeuronFactory(
                        randomizer)));

            var mlp = mlpFactory.CreateMLP(
                string.Format(
                    "mlp{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss")),
                new IFunction[]
                {
                    null,
                    new RLUFunction(), 
                    new RLUFunction(), 
                    new RLUFunction(), 
                    new SigmoidFunction(1f), 
                },
                new int[]
                {
                    validationData.InputLength,
                    800,
                    100,
                    800,
                    validationData.OutputLength 
                });

            ConsoleAmbientContext.Console.WriteLine("Created " + mlp.GetLayerInformation());

            var rootContainer = new SavelessArtifactContainer(
                ".",
                serialization);

            var validation = new Validation(
                new ClassificationAccuracyCalculator(
                    new HalfSquaredEuclidianDistance(), 
                    validationData), 
                null
                );

            const int epocheCount = 1;

            var config = new LearningAlgorithmConfig(
                new HalfSquaredEuclidianDistance(),
                new ConstLearningRate(1/128f), 
                2,
                0f,
                epocheCount,
                -1f,
                -1f
                );

            var mlpFolderName = string.Format(
                "mlp{0}",
                DateTime.Now.ToString("yyyyMMddHHmmss"));

            var mlpContainer = rootContainer.GetChildContainer(mlpFolderName);

            var mlpContainerHelper = new MLPContainerHelper();

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(false), true))
            {
                var maskContainerFactory = new BigArrayMaskContainerFactory(
                    new ConstRandomizer(0.5f, 5),
                    clProvider
                    );

                var inferencerFactory = new GPULayerInferencerFactory(
                    new ConstRandomizer(0.5f, 3), 
                    clProvider,
                    1,
                    0.5f
                    );

                if (useNew)
                {
                    var algo = new Backpropagation(
                        new DropConnect2EpocheTrainer(
                            mlp,
                            config,
                            maskContainerFactory,
                            inferencerFactory,
                            clProvider),
                        mlpContainerHelper,
                        mlpContainer,
                        mlp,
                        validation,
                        config
                        );

                    var acc = algo.Train(
                        trainDataSetProvider
                        );

                    return acc.PerItemError;
                }
                else
                {
                    var algo = new Backpropagation(
                        new DropConnectEpocheTrainer(
                            mlp,
                            config,
                            maskContainerFactory,
                            inferencerFactory,
                            clProvider),
                        mlpContainerHelper,
                        mlpContainer,
                        mlp,
                        validation,
                        config
                        );

                    var acc = algo.Train(
                        trainDataSetProvider
                        );

                    return acc.PerItemError;
                }
            }
        }

        private static IDataSetProvider GetTrainProvider(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new MNISTDataItemLoaderForDelete(
                "_MNIST_DATABASE/mnist/trainingset/",
                maxCountFilesInCategory,
                false,
                dataItemFactory,
                new DefaultNormalizer()
                );

            if (isNeedToGNormalize)
            {
                dataItemLoader.GNormalize();
            }

            if (isNeedToNormalize)
            {
                dataItemLoader.Normalize();
            }

            var iteratorFactory = new DataIteratorFactory();

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        //new ToAutoencoderDataItemTransformation(dataItemFactory);
                        new NoConvertDataItemTransformation();
                });

            var dataSetFactory = new DataSetFactory(
                iteratorFactory,
                itemTransformationFactory
                );

            var dataSetProvider = new DataSetProvider(
                dataSetFactory,
                dataItemLoader
                );

            return
                dataSetProvider;
        }

        private static IDataSet GetValidation(
            int maxCountFilesInCategory,
            bool isNeedToGNormalize,
            bool isNeedToNormalize
            )
        {
            var dataItemFactory = new DataItemFactory();

            var dataItemLoader = new MNISTDataItemLoaderForDelete(
                "_MNIST_DATABASE/mnist/testset/",
                maxCountFilesInCategory,
                false,
                dataItemFactory,
                new DefaultNormalizer()
                );

            if (isNeedToGNormalize)
            {
                dataItemLoader.GNormalize();
            }

            if (isNeedToNormalize)
            {
                dataItemLoader.Normalize();
            }

            var iterationFactory = new DataIteratorFactory(
                );

            var itemTransformationFactory = new DataItemTransformationFactory(
                (epochNumber) =>
                {
                    return
                        //new ToAutoencoderDataItemTransformation(dataItemFactory);
                        new NoConvertDataItemTransformation();
                });

            var validationDataSet = new DataSet(
                iterationFactory,
                itemTransformationFactory,
                dataItemLoader,
                0
                );

            return
                validationDataSet;
        }

    }


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
