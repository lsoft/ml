using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using MyNN.BoltzmannMachines.BinaryBinary.DBN;
using MyNN.BoltzmannMachines.DBNInfo;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Layer.Factory.WeightLoader;
using MyNN.MLP2.Structure.Neurons;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.OutputConsole;
using MyNN.Randomizer;

namespace MyNN.MLP2.Structure.Factory
{
    public class MLPFactory : IMLPFactory
    {
        private readonly ILayerFactory _layerFactory;

        public MLPFactory(
            ILayerFactory layerFactory
            )
        {
            if (layerFactory == null)
            {
                throw new ArgumentNullException("layerFactory");
            }

            _layerFactory = layerFactory;
        }

        public IMLP CreateMLP(
            string root,
            string folderName, 
            IFunction[] activationFunction, 
            params int[] neuronCountList)
        {
            //root, folderName  allowed to be null

            if (neuronCountList == null)
            {
                throw new ArgumentNullException("neuronCountList");
            }
            if (activationFunction == null || activationFunction.Length != neuronCountList.Length)
            {
                throw new InvalidOperationException("activationFunction == null || activationFunction.Length != neuronCountList.Length");
            }

            //формируем слои
            var layerList = new ILayer[neuronCountList.Length];

            //создаем входной слой
            layerList[0] = _layerFactory.CreateInputLayer(neuronCountList[0]);

            //создаем скрытые слои и выходной слой
            var isPreviousLayerHadBiasNeuron = true;
            for (var cc = 1; cc < neuronCountList.Length; cc++)
            {
                var isLayerHasBiasNeuron = cc != (neuronCountList.Length - 1);

                layerList[cc] = _layerFactory.CreateLayer(
                    activationFunction[cc],
                    neuronCountList[cc],
                    neuronCountList[cc - 1],
                    isLayerHasBiasNeuron,
                    isPreviousLayerHadBiasNeuron
                    );

                isPreviousLayerHadBiasNeuron = isLayerHasBiasNeuron;
            }

            return
                new MLP(
                    _layerFactory,
                    root,
                    folderName,
                    layerList);
        }

        public IMLP CreateMLP(
            string root,
            string folderName, 
            ILayer[] layerList
            )
        {
            //root, folderName allowed to be null

            if (layerList == null)
            {
                throw new ArgumentNullException("layerList");
            }

            return
                new MLP(
                    _layerFactory,
                    root,
                    folderName,
                    layerList);
        }

        public IMLP CreateMLP(
            IDBNInformation dbnInformation,
            string root,
            string folderName,
            IFunction[] activationFunction
            )
        {
            if (dbnInformation == null)
            {
                throw new ArgumentNullException("dbnInformation");
            }
            //root, folderName allowed to be null
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            #region проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

            if (dbnInformation.LayerCount != activationFunction.Length)
            {
                throw new InvalidOperationException(
                    string.Format(
                        "Layer count from dbn.info {0} != layer count from parameters {1}",
                        dbnInformation.LayerCount,
                        activationFunction.Length));
            }

            #endregion

            #region создаем слои

            ConsoleAmbientContext.Console.WriteLine("Load weights from DBN...");

            var layerList = new ILayer[activationFunction.Length];

            layerList[0] = _layerFactory.CreateInputLayer(dbnInformation.LayerSizes[0]);

            for (var layerIndex = 1; layerIndex <= Math.Min(layerList.Length, dbnInformation.LayerCount); layerIndex++)
            {
                var isLayerHasBiasNeuron = layerIndex != (dbnInformation.LayerCount - 1);

                //создаем слой
                var layer = _layerFactory.CreateLayer(
                    activationFunction[layerIndex],
                    dbnInformation.LayerSizes[layerIndex],
                    layerList[layerIndex - 1].NonBiasNeuronCount,
                    isLayerHasBiasNeuron,
                    layerList[layerIndex - 1].IsBiasNeuronExists
                    );

                //загружаем веса
                IWeightLoader weightLoader;
                dbnInformation.GetWeightLoaderForLayer(layerIndex, out weightLoader);
                weightLoader.LoadWeights(layer);

                layerList[layerIndex] = layer;
            }

            ConsoleAmbientContext.Console.WriteLine("Load weights done");

            #endregion

            //создаем MLP
            var mlp =
                new MLP(
                    _layerFactory,
                    root,
                    folderName,
                    layerList);

            return
                mlp;
        }

        //public IMLP CreateMLP(
        //    string dbnInfoRoot,
        //    string root,
        //    string folderName, 
        //    IFunction[] activationFunction
        //    )
        //{
        //    if (dbnInfoRoot == null)
        //    {
        //        throw new ArgumentNullException("dbnInfoRoot");
        //    }
        //    //root, folderName allowed to be null
        //    if (activationFunction == null)
        //    {
        //        throw new ArgumentNullException("activationFunction");
        //    }

        //    //количество слоев в обученной DBN
        //    var dbnFolderCount =
        //        Directory.GetDirectories(dbnInfoRoot)
        //        .ToList()
        //        .FindAll(j => Path.GetFileName(j).StartsWith(DeepBeliefNetwork.RbmFolderName))
        //        .Count;

        //    #region проверяем наличие файла dbn.info и содержимое в нем

        //    var pathToDbnInfo = Path.Combine(dbnInfoRoot, "dbn.info");
        //    if (!File.Exists(pathToDbnInfo))
        //    {
        //        throw new InvalidOperationException("dbn.info does not found, information about MLP structure does not exits.");
        //    }

        //    ConsoleAmbientContext.Console.WriteLine("dbn.info found...");

        //    #endregion

        //    #region проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

        //    var layerSizes = FileDBNInformation.ExtractLayersSizeFromDbnInfo(pathToDbnInfo);

        //    if (layerSizes == null || layerSizes.Count == 0)
        //    {
        //        throw new InvalidOperationException("layer sizes are empty, information about MLP structure does not exits.");
        //    }
        //    if (layerSizes.Count != activationFunction.Length)
        //    {
        //        throw new InvalidOperationException(
        //            string.Format(
        //                "Layer count from dbn.info {0} != layer count from parameters {1}",
        //                layerSizes.Count,
        //                activationFunction.Length));
        //    }

        //    #endregion

        //    #region создаем слои

        //    ConsoleAmbientContext.Console.WriteLine("Load weights from DBN...");

        //    var layerList = new ILayer[activationFunction.Length];
            
        //    layerList[0] = _layerFactory.CreateInputLayer(layerSizes[0]);

        //    for (var layerIndex = 1; layerIndex <= Math.Min(layerList.Length, dbnFolderCount); layerIndex++)
        //    {
        //        var rbmFolderPath = Path.Combine(dbnInfoRoot, DeepBeliefNetwork.RbmFolderName + (layerIndex - 1));
                
        //        var lastEpocheNumber =
        //            (from d in Directory.EnumerateDirectories(rbmFolderPath)
        //             orderby this.ExtractRBMEpocheNumberFromDirName(d) descending
        //             select d).First();

        //        ConsoleAmbientContext.Console.WriteLine(
        //            string.Format(
        //                "Processed layers {0} weights from: {1}",
        //                layerIndex,
        //                lastEpocheNumber));

        //        var pathToWeightsFile = Path.Combine(lastEpocheNumber, "weights.bin"); //!!! заменить юзанием конст

        //        var isLayerHasBiasNeuron = layerIndex != (layerSizes.Count - 1);

        //        //создаем слой
        //        var layer = _layerFactory.CreateLayer(
        //            activationFunction[layerIndex],
        //            layerSizes[layerIndex],
        //            layerList[layerIndex - 1].NonBiasNeuronCount,
        //            isLayerHasBiasNeuron,
        //            layerList[layerIndex - 1].IsBiasNeuronExists
        //            );

        //        //загружаем веса
        //        var weightLoader = new RBMWeightLoader(pathToWeightsFile);
        //        weightLoader.LoadWeights(layer);

        //        layerList[layerIndex] = layer;
        //    }

        //    ConsoleAmbientContext.Console.WriteLine("Load weights done");

        //    #endregion

        //    //создаем MLP
        //    var mlp = 
        //        new MLP(
        //            _layerFactory,
        //            root,
        //            folderName,
        //            layerList);

        //    return
        //        mlp;
        //}

        public IMLP CreateAutoencoderMLP(
            IDBNInformation dbnInformation,
            string root,
            string folderName,
            IFunction[] activationFunction
            )
        {
            if (dbnInformation == null)
            {
                throw new ArgumentNullException("dbnInformation");
            }
            //root, folderName allowed to be null
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            #region проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

            //проверяем что количество слоев в DBN и количество слоев в MLP сочетаемо
            var mlpLayersCount = activationFunction.Length;
            var autoencoderLayers = (mlpLayersCount - 1) / 2;
            if (dbnInformation.LayerCount > autoencoderLayers)
            {
                throw new InvalidOperationException("dbn.info contains more layers than can be gets by this autoencoder, cancel operation");
            }

            #endregion

            #region создаем слои

            ConsoleAmbientContext.Console.WriteLine("Load weights from DBN...");

            var layerList = new ILayer[activationFunction.Length];

            layerList[0] = _layerFactory.CreateInputLayer(dbnInformation.LayerSizes[0]);

            for (var layerIndex = 1; layerIndex <= Math.Min(layerList.Length, dbnInformation.LayerCount); layerIndex++)
            {
                IWeightLoader encoderWeightLoader, decoderWeightLoader;
                dbnInformation.GetAutoencoderWeightLoaderForLayer(
                    layerIndex,
                    mlpLayersCount,
                    out encoderWeightLoader,
                    out decoderWeightLoader);

                //создаем слой кодирования
                {
                    var encoderLayer = _layerFactory.CreateLayer(
                        activationFunction[layerIndex],
                        dbnInformation.LayerSizes[layerIndex],
                        layerList[layerIndex - 1].NonBiasNeuronCount,
                        true,
                        layerList[layerIndex - 1].IsBiasNeuronExists
                        );

                    //загружаем веса
                    encoderWeightLoader.LoadWeights(encoderLayer);

                    layerList[layerIndex] = encoderLayer;
                }

                //создаем слой декодирования
                {
                    var decoderLayerIndex = mlpLayersCount - layerIndex;
                    var isDecoderLayerHasBiasNeuron = decoderLayerIndex != (activationFunction.Length - 1);

                    var decoderLayer = _layerFactory.CreateLayer(
                        activationFunction[decoderLayerIndex],
                        layerList[layerIndex - 1].NonBiasNeuronCount,
                        layerList[layerIndex].NonBiasNeuronCount,
                        isDecoderLayerHasBiasNeuron,
                        true
                        );

                    //загружаем веса
                    decoderWeightLoader.LoadWeights(decoderLayer);

                    layerList[decoderLayerIndex] = decoderLayer;
                }
            }

            ConsoleAmbientContext.Console.WriteLine("Load weights done");

            #endregion

            //создаем MLP
            var mlp =
                new MLP(
                    _layerFactory,
                    root,
                    folderName,
                    layerList);

            return
                mlp;
        }

        //public IMLP CreateAutoencoderMLP(
        //    string dbnInfoRoot,
        //    string root,
        //    string folderName,
        //    IFunction[] activationFunction
        //    )
        //{
        //    if (dbnInfoRoot == null)
        //    {
        //        throw new ArgumentNullException("dbnInfoRoot");
        //    }
        //    //root, folderName allowed to be null
        //    if (activationFunction == null)
        //    {
        //        throw new ArgumentNullException("activationFunction");
        //    }

        //    //количество слоев в обученной DBN
        //    var dbnFolderCount =
        //        Directory.GetDirectories(dbnInfoRoot)
        //            .ToList()
        //            .FindAll(j => Path.GetFileName(j).StartsWith(DeepBeliefNetwork.RbmFolderName))
        //            .Count;

        //    #region проверяем наличие файла dbn.info и содержимое в нем

        //    var pathToDbnInfo = Path.Combine(dbnInfoRoot, "dbn.info");
        //    if (!File.Exists(pathToDbnInfo))
        //    {
        //        throw new InvalidOperationException("dbn.info does not found, information about MLP structure does not exits.");
        //    }

        //    ConsoleAmbientContext.Console.WriteLine("dbn.info found...");

        //    #endregion

        //    #region проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

        //    var layerSizes = FileDBNInformation.ExtractLayersSizeFromDbnInfo(pathToDbnInfo);

        //    if (layerSizes == null || layerSizes.Count == 0)
        //    {
        //        throw new InvalidOperationException("layer sizes are empty, information about MLP structure does not exits.");
        //    }

        //    //проверяем что количество слоев в DBN и количество слоев в MLP сочетаемо
        //    var mlpLayersCount = activationFunction.Length;
        //    var autoencoderLayers = (mlpLayersCount - 1)/2;
        //    if (dbnFolderCount > autoencoderLayers)
        //    {
        //        throw new InvalidOperationException("dbn.info contains more layers than can be gets by this autoencoder, cancel operation");
        //    }

        //    ConsoleAmbientContext.Console.WriteLine("dbn.info checks ok...");

        //    #endregion

        //    #region создаем слои

        //    ConsoleAmbientContext.Console.WriteLine("Load weights from DBN...");

        //    var layerList = new ILayer[activationFunction.Length];

        //    layerList[0] = _layerFactory.CreateInputLayer(layerSizes[0]);

        //    for (var layerIndex = 1; layerIndex <= Math.Min(layerList.Length, dbnFolderCount); layerIndex++)
        //    {
        //        var rbmFolderPath = Path.Combine(dbnInfoRoot, DeepBeliefNetwork.RbmFolderName + (layerIndex - 1));

        //        var lastEpocheNumber =
        //            (from d in Directory.EnumerateDirectories(rbmFolderPath)
        //             orderby this.ExtractRBMEpocheNumberFromDirName(d) descending
        //             select d).First();

        //        ConsoleAmbientContext.Console.WriteLine(
        //            string.Format(
        //                "Processed layers {0}, {1} weights from: {2}",
        //                layerIndex,
        //                (mlpLayersCount - layerIndex),
        //                lastEpocheNumber));

        //        //var pathToWeightsFile = Path.Combine(lastEpocheNumber, "weights.bin");
        //        //layerList[layerIndex].LoadWeightsFromRBM(pathToWeightsFile);
        //        //layerList[mlpLayersCount - layerIndex].LoadAutoencoderWeightsFromRBM(pathToWeightsFile);

        //        var pathToWeightsFile = Path.Combine(lastEpocheNumber, "weights.bin"); //!!! заменить юзанием конст

        //        //создаем слой кодирования
        //        {
        //            var encoderLayer = _layerFactory.CreateLayer(
        //                activationFunction[layerIndex],
        //                layerSizes[layerIndex],
        //                layerList[layerIndex - 1].NonBiasNeuronCount,
        //                true,
        //                layerList[layerIndex - 1].IsBiasNeuronExists
        //                );

        //            //загружаем веса
        //            var weightLoader = new RBMWeightLoader(pathToWeightsFile);
        //            weightLoader.LoadWeights(encoderLayer);

        //            layerList[layerIndex] = encoderLayer;
        //        }

        //        //создаем слой декодирования
        //        {
        //            var decoderLayerIndex = mlpLayersCount - layerIndex;
        //            var isDecoderLayerHasBiasNeuron = decoderLayerIndex != (activationFunction.Length - 1);

        //            var decoderLayer = _layerFactory.CreateLayer(
        //                activationFunction[decoderLayerIndex],
        //                layerList[layerIndex - 1].NonBiasNeuronCount,
        //                layerList[layerIndex].NonBiasNeuronCount,
        //                isDecoderLayerHasBiasNeuron,
        //                true
        //                );

        //            //загружаем веса
        //            var weightLoader = new RBMAutoencoderWeightLoader(pathToWeightsFile);
        //            weightLoader.LoadWeights(decoderLayer);

        //            layerList[decoderLayerIndex] = decoderLayer;
        //        }
        //    }

        //    ConsoleAmbientContext.Console.WriteLine("Load weights done");

        //    #endregion

        //    //создаем MLP
        //    var mlp =
        //        new MLP(
        //            _layerFactory,
        //            root,
        //            folderName,
        //            layerList);

        //    return
        //        mlp;
        //}

        private int ExtractRBMEpocheNumberFromDirName(string dirname)
        {
            var lastSlashIndex = dirname.ToList().FindLastIndex(j => j == '\\');

            var stringNumber = dirname.Substring(lastSlashIndex + 8); //!!! константу 8 заменить на ссылку на имя.Length

            return int.Parse(stringNumber);
        }

    }
}