using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.IO;
using System.Linq;
using MyNN.BoltzmannMachines.BinaryBinary.DBN;
using MyNN.Data;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.OutputConsole;
using MyNN.Randomizer;

namespace MyNN.MLP2.Structure
{
    [Serializable]
    public class MLP : IMLP
    {
        [NonSerialized]
        private string _root;
        public string Root
        {
            get
            {
                if (string.IsNullOrEmpty(_root))
                {
                    _root = ".";
                }

                return _root;
            }

            private set
            {
                _root = value;
            }
        }

        [NonSerialized]
        private string _folderName;
        public string FolderName
        {
            get
            {
                if (string.IsNullOrEmpty(_folderName))
                {
                    _folderName = "MLP" + DateTime.Now.ToString("yyyyMMddHHmmss");
                }

                return
                    _folderName;
            }

            private set
            {
                _folderName = value;
            }
        }

        public string WorkFolderPath
        {
            get
            {
                return
                    Path.Combine(Root, FolderName);
            }
        }

        private readonly ILayerFactory _layerFactory;
        private volatile ILayer[] _layers;
        public ILayer[] Layers
        {
            get
            {
                return
                    _layers;
            }
        }

        public MLP(
            ILayerFactory layerFactory,
            string root, 
            string folderName,
            ILayer[] layerList)
        {
            if (layerFactory == null)
            {
                throw new ArgumentNullException("layerFactory");
            }
            //root, folderName allowed to be null
            if (layerList == null)
            {
                throw new ArgumentNullException("layerList");
            }

            _layerFactory = layerFactory;

            Root = root;
            FolderName = folderName;
            
            //формируем слои
            this._layers = layerList;

            this.CreateWorkFolderFolder();
        }

        public void SetRootFolder(string root)
        {
            if (root == null)
            {
                throw new ArgumentNullException("root");
            }

            this.Root = root;
        }

        private void CreateWorkFolderFolder()
        {
            var p = WorkFolderPath;

            if (!Directory.Exists(p))
            {
                Directory.CreateDirectory(p);
            }
        }

        public string GetLayerInformation()
        {
            return
                string.Join(" -> ", this.Layers.ToList().ConvertAll(j => j.GetLayerInformation()));
        }

        /// <summary>
        /// Обрезать автоенкодер. Удаляются слои, начиная с узкого слоя и до конца
        /// </summary>
        public void AutoencoderCutTail()
        {
            var lls = new ILayer[(this.Layers.Length + 1) / 2];
            Array.Copy(this.Layers, 0, lls, 0, lls.Length);

            this._layers = lls;

            //у последнего слоя убираем Bias нейрон
            var nll = this.Layers.Last();
            nll.RemoveBiasNeuron();
        }

        /// <summary>
        /// Убрать последний слой
        /// </summary>
        public void CutLastLayer()
        {
            var lls = new ILayer[this.Layers.Length - 1];
            Array.Copy(this.Layers, 0, lls, 0, lls.Length);

            this._layers = lls;
        }


        /// <summary>
        /// Обрезать автоенкодер. Удаляются слои, начиная с первого и до узкого слоя
        /// </summary>
        public void AutoencoderCutHead()
        {
            var lls = new ILayer[(this.Layers.Length + 1) / 2];
            Array.Copy(this.Layers, this.Layers.Length - lls.Length, lls, 0, lls.Length);

            this._layers = lls;
        }

        public void AddLayer(
            IFunction activationFunction,
            int nonBiasNeuronCount,
            bool isNeedBiasNeuron)
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            var lastl = this._layers[this._layers.Length - 1];
            if (!lastl.IsBiasNeuronExists)
            {
                lastl.AddBiasNeuron();
            }

            var newl = new ILayer[this._layers.Length + 1];
            this._layers.CopyTo(newl, 0);

            var bornLayer = _layerFactory.CreateLayer(
                activationFunction,
                nonBiasNeuronCount,
                lastl.NonBiasNeuronCount,
                isNeedBiasNeuron,
                true
                );
            newl[this._layers.Length] = bornLayer;

            this._layers = newl;
        }

        //!!! извлечь в специальный вид фабрики IMLPFromDBNFactory
        /*
        #region relation with deep belief network

        /// <summary>
        /// Загружаем веса из предобученной DBN
        /// </summary>
        /// <param name="root">Путь к папке, где обучена DBN</param>
        public void LoadWeightsFromDBN(
            string root)
        {
            //количество слоев в обученной DBN
            var dbnFolderCount =
                Directory.GetDirectories(root)
                .ToList()
                .FindAll(j => Path.GetFileName(j).StartsWith(DeepBeliefNetwork.RbmFolderName))
                .Count;

            #region проверяем наличие файла dbn.info и содержимое в нем

            var pathToDbnInfo = Path.Combine(root, "dbn.info");
            if (File.Exists(pathToDbnInfo))
            {
                ConsoleAmbientContext.Console.WriteLine("dbn.info found, check processed...");

                //проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

                var layerSizes = DeepBeliefNetwork.ExtractLayersSizeFromDbnInfo(pathToDbnInfo);

                if (layerSizes != null && layerSizes.Count > 0)
                {
                    for (var cc = 0; cc <= Math.Min(this.Layers.Length, dbnFolderCount); cc++)
                    {
                        if (layerSizes[cc] != this.Layers[cc].NonBiasNeuronCount)
                        {
                            ConsoleAmbientContext.Console.WriteLine(
                                string.Format(
                                    "Layer {0} has different neuron count: {1} in MLP vs {2} in DBN, weights loading fails",
                                    cc,
                                    this.Layers[cc].NonBiasNeuronCount,
                                    layerSizes[cc]));
                            return;
                        }
                    }

                    ConsoleAmbientContext.Console.WriteLine("dbn.info checks ok...");
                }
                else
                {
                    ConsoleAmbientContext.Console.WriteLine("dbn.info can't be parsed, check skipped...");
                }
            }
            else
            {
                ConsoleAmbientContext.Console.WriteLine("dbn.info not found, check skipped...");
            }

            #endregion


            ConsoleAmbientContext.Console.WriteLine("Load weights from DBN...");

            for (var layerIndex = 1; layerIndex <= Math.Min(this.Layers.Length, dbnFolderCount); layerIndex++)
            {
                var layer = Path.Combine(root, DeepBeliefNetwork.RbmFolderName + (layerIndex - 1));
                var lastEpoche =
                    (from d in Directory.EnumerateDirectories(layer)
                     orderby this.ExtractRBMEpocheNumberFromDirName(d) descending
                     select d).First();

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Processed layers {0} weights from: {1}",
                        layerIndex,
                        lastEpoche));


                var pathToWeightsFile = Path.Combine(lastEpoche, "weights.bin");
                this.Layers[layerIndex].LoadWeightsFromRBM(pathToWeightsFile);
            }

            ConsoleAmbientContext.Console.WriteLine("Load weights done");
        }

        /// <summary>
        /// Загружаем веса в MLP-автоенкодер из предобученной DBN
        /// </summary>
        /// <param name="root">Путь к папке, где обучена DBN</param>
        public void LoadAutoencoderWeightsFromDBN(
            string root)
        {
            //количество слоев в обученной DBN
            var dbnFolderCount =
                Directory.GetDirectories(root)
                .ToList()
                .FindAll(j => Path.GetFileName(j).StartsWith(DeepBeliefNetwork.RbmFolderName))
                .Count;

            var mlpLayersCount = this.Layers.Length;

            #region проверяем наличие файла dbn.info и содержимое в нем

            var pathToDbnInfo = Path.Combine(root, "dbn.info");
            if (File.Exists(pathToDbnInfo))
            {
                ConsoleAmbientContext.Console.WriteLine("dbn.info found, check processed...");

                //проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

                var layerSizes = DeepBeliefNetwork.ExtractLayersSizeFromDbnInfo(pathToDbnInfo);

                if (layerSizes != null && layerSizes.Count > 0)
                {
                    //проверяем что количество слоев в DBN и количество слоев в MLP сочетаемо
                    var autoencoderLayers = (mlpLayersCount - 1) / 2;
                    if (dbnFolderCount <= autoencoderLayers)
                    {
                        for (var cc = 0; cc <= Math.Min(autoencoderLayers, dbnFolderCount); cc++)
                        {
                            if (layerSizes[cc] != this.Layers[cc].NonBiasNeuronCount)
                            {
                                ConsoleAmbientContext.Console.WriteLine(
                                    string.Format(
                                        "Layer {0} has different neuron count: {1} in MLP vs {2} in DBN, cancel operation",
                                        cc,
                                        this.Layers[cc].NonBiasNeuronCount,
                                        layerSizes[cc]));
                                return;
                            }

                            if (layerSizes[cc] != this.Layers[mlpLayersCount - cc - 1].NonBiasNeuronCount)
                            {
                                ConsoleAmbientContext.Console.WriteLine(
                                    string.Format(
                                        "Layer {0} has different neuron count: {1} in MLP vs {2} in DBN, cancel operation",
                                        (mlpLayersCount - cc - 1),
                                        this.Layers[mlpLayersCount - cc - 1].NonBiasNeuronCount,
                                        layerSizes[cc]));
                                return;
                            }
                        }

                        ConsoleAmbientContext.Console.WriteLine("dbn.info checks ok...");
                    }
                    else
                    {
                        ConsoleAmbientContext.Console.WriteLine("dbn contains more layer than can be gets by this autoencoder, cancel operation");
                    }
                }
                else
                {
                    ConsoleAmbientContext.Console.WriteLine("dbn.info can't be parsed, check skipped...");
                }
            }
            else
            {
                ConsoleAmbientContext.Console.WriteLine("dbn.info not found, check skipped...");
            }

            #endregion


            ConsoleAmbientContext.Console.WriteLine("Load weights from DBN...");

            for (var layerIndex = 1; layerIndex <= Math.Min(this.Layers.Length, dbnFolderCount); layerIndex++)
            {
                var layer = Path.Combine(root, DeepBeliefNetwork.RbmFolderName + (layerIndex - 1));
                var lastEpoche =
                    (from d in Directory.EnumerateDirectories(layer)
                     orderby this.ExtractRBMEpocheNumberFromDirName(d) descending
                     select d).First();

                ConsoleAmbientContext.Console.WriteLine(
                    string.Format(
                        "Processed layers {0}, {1} weights from: {2}",
                        layerIndex,
                        (mlpLayersCount - layerIndex),
                        lastEpoche));

                var pathToWeightsFile = Path.Combine(lastEpoche, "weights.bin");

                this.Layers[layerIndex].LoadWeightsFromRBM(pathToWeightsFile);
                this.Layers[mlpLayersCount - layerIndex].LoadAutoencoderWeightsFromRBM(pathToWeightsFile);
            }

            ConsoleAmbientContext.Console.WriteLine("Load weights done");
        }

        private int ExtractRBMEpocheNumberFromDirName(string dirname)
        {
            var lastSlashIndex = dirname.ToList().FindLastIndex(j => j == '\\');

            var stringNumber = dirname.Substring(lastSlashIndex + 8);

            return int.Parse(stringNumber);
        }

        #endregion
        //*/
    }
}
