using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MyNN.BoltzmannMachines.BinaryBinary.DBN;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons;
using MyNN.NeuralNet.Structure.Neurons.Function;

namespace MyNN.NeuralNet.Structure
{
    [Serializable]
    public class MultiLayerNeuralNetwork 
        //: ITrainMultiLayerNetwork<Layer>
    {
        [NonSerialized]
        private IMultilayerComputer _computer;
        public IMultilayerComputer Computer
        {
            get
            {
                return
                    _computer;
            }

            private set
            {
                _computer = value;
            }
        }

        public Layer[] Layers
        {
            get;
             set;
        }

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

        public MultiLayerNeuralNetwork(
            string root,
            string folderName,
            Layer[] layers)
        {
            //root, folderName  allowed to be null

            if (layers == null)
            {
                throw new ArgumentNullException("layers");
            }

            Root = root;
            FolderName = folderName;

            //формируем слои
            this.Layers = layers;
        }

        public MultiLayerNeuralNetwork(
            string root,
            string folderName,
            IFunction[] activationFunction,
            ref int seed,
            params int[] neuronCountList)
        {
            #region validate

            //root, folderName  allowed to be null

            if (activationFunction == null || activationFunction.Length != neuronCountList.Length)
            {
                throw new InvalidOperationException("activationFunction == null || activationFunction.Length != neuronCountList.Length");
            }

            #endregion

            Root = root;
            FolderName = folderName;

            //формируем слои
            this.Layers = new Layer[neuronCountList.Length];

            //создаем входной слой
            this.Layers[0] = new Layer(
                neuronCountList[0],
                true);

            //создаем скрытые слои и выходной слой
            var isPreviousLayerHadBiasNeuron = true;
            for (var cc = 1; cc < neuronCountList.Length; cc++)
            {
                var isLayerHasBiasNeuron = cc != (neuronCountList.Length - 1);

                this.Layers[cc] = new Layer(
                    activationFunction[cc],
                    neuronCountList[cc],
                    neuronCountList[cc - 1],
                    isLayerHasBiasNeuron,
                    isPreviousLayerHadBiasNeuron,
                    ref seed);

                isPreviousLayerHadBiasNeuron = isLayerHasBiasNeuron;
            }

            CreateWorkFolderFolder();
        }

        public MultiLayerNeuralNetwork(
            string root,
            string folderName,
            IFunction activationFunction,
            ref int seed,
            params int[] neuronCountList)
        {
            #region validate

            //root, folderName  allowed to be null

            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            #endregion

            Root = root;
            FolderName = folderName;

            //формируем слои
            this.Layers = new Layer[neuronCountList.Length];

            //создаем входной слой
            this.Layers[0] = new Layer(
                neuronCountList[0],
                true);

            //создаем скрытые слои и выходной слой
            var isPreviousLayerHadBiasNeuron = true;
            for (var cc = 1; cc < neuronCountList.Length; cc++)
            {
                var isLayerHasBiasNeuron = cc != (neuronCountList.Length - 1);

                this.Layers[cc] = new Layer(
                    activationFunction,
                    neuronCountList[cc],
                    neuronCountList[cc - 1],
                    isLayerHasBiasNeuron,
                    isPreviousLayerHadBiasNeuron,
                    ref seed);

                isPreviousLayerHadBiasNeuron = isLayerHasBiasNeuron;
            }

            CreateWorkFolderFolder();
        }
        
        public MultiLayerNeuralNetwork(
            string root,
            string folderName,
            ref int seed,
            params int[] neuronCountList)
            : this(
                root,
                folderName,
                new SigmoidFunction(1),
                ref seed,
                neuronCountList)
        {
            
        }

        private void CreateWorkFolderFolder()
        {
            var p = WorkFolderPath;

            if (!Directory.Exists(p))
            {
                Directory.CreateDirectory(p);
            }
        }

        public void SetComputer(IMultilayerComputer computer)
        {
            Computer = computer;
        }

        public float[] ComputeOutput(float[] inputVector)
        {
            return
                Computer.ComputeOutput(inputVector);
        }

        public List<float[]> ComputeOutput(List<float[]> inputVectors)
        {
            return
                Computer.ComputeOutput(inputVectors);
        }

        public void ExecuteComputation()
        {
            Computer.ExecuteComputation();
        }


        public void AutoencoderCut()
        {
            var fl = this.Layers.First();
            var ll = this.Layers.Last();

            if (fl.NonBiasNeuronCount != ll.NonBiasNeuronCount)
            {
                throw new InvalidOperationException("Это не автоенкодер");
            }

            var lls = new Layer[(this.Layers.Length + 1) / 2];
            Array.Copy(this.Layers, 0, lls, 0, lls.Length);

            this.Layers = lls;

            //у последнего слоя убираем Bias нейрон
            var nll = this.Layers.Last();
            nll.RemoveBiasNeuron();
        }

        public void AddLayer(
            IFunction activationFunction,
            int nonBiasNeuronCount,
            bool isNeedBiasNeuron,
            ref int rndSeed)
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            var lastl = this.Layers[this.Layers.Length - 1];

            var newl = new Layer[this.Layers.Length + 1];
            this.Layers.CopyTo(newl, 0);

            newl[this.Layers.Length] = new Layer(
                activationFunction,
                nonBiasNeuronCount,
                lastl.NonBiasNeuronCount,
                isNeedBiasNeuron,
                (lastl.NonBiasNeuronCount + 1) == lastl.Neurons.Length,
                ref rndSeed);

            this.Layers = newl;
        }

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
                .FindAll(j => Path.GetFileName(j).StartsWith("rbm_layer"))
                .Count;

            #region проверяем наличие файла dbn.info и содержимое в нем

            var pathToDbnInfo = Path.Combine(root, "dbn.info");
            if (File.Exists(pathToDbnInfo))
            {
                Console.WriteLine("dbn.info found, check processed...");

                //проверяем что размеры слоев в DBN сходятся с размерами слоев в MLP

                var layerSizes = DeepBeliefNetwork.ExtractLayersSizeFromDbnInfo(pathToDbnInfo);

                if (layerSizes != null && layerSizes.Count > 0)
                {
                    for (var cc = 0; cc <= Math.Min(this.Layers.Length, dbnFolderCount); cc++)
                    {
                        if (layerSizes[cc] != this.Layers[cc].NonBiasNeuronCount)
                        {
                            Console.WriteLine(
                                string.Format(
                                    "Layer {0} has different neuron count: {1} in MLP vs {2} in DBN, weights loading fails",
                                    cc,
                                    this.Layers[cc].NonBiasNeuronCount,
                                    layerSizes[cc]));
                            return;
                        }
                    }

                    Console.WriteLine("dbn.info checks ok...");
                }
                else
                {
                    Console.WriteLine("dbn.info can't be parsed, check skipped...");
                }
            }
            else
            {
                Console.WriteLine("dbn.info not found, check skipped...");
            }

            #endregion


            Console.WriteLine("Load weights from DBN...");

            for (var layerIndex = 1; layerIndex <= Math.Min(this.Layers.Length, dbnFolderCount); layerIndex++)
            {
                var layer = Path.Combine(root, "rbm_layer" + (layerIndex - 1));
                var lastEpoche =
                    (from d in Directory.EnumerateDirectories(layer)
                     orderby this.ExtractRBMEpocheNumberFromDirName(d) descending
                     select d).First();

                Console.WriteLine(
                    string.Format(
                        "Processed layers {0} weights from: {1}",
                        layerIndex,
                        lastEpoche));


                var pathToWeightsFile = Path.Combine(lastEpoche, "weights.bin");
                this.Layers[layerIndex].LoadWeightsFromRBM(pathToWeightsFile);
            }

            Console.WriteLine("Load weights done");
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
                .FindAll(j => Path.GetFileName(j).StartsWith("rbm_layer"))
                .Count;

            var mlpLayersCount = this.Layers.Length;

            #region проверяем наличие файла dbn.info и содержимое в нем

            var pathToDbnInfo = Path.Combine(root, "dbn.info");
            if (File.Exists(pathToDbnInfo))
            {
                Console.WriteLine("dbn.info found, check processed...");

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
                                Console.WriteLine(
                                    string.Format(
                                        "Layer {0} has different neuron count: {1} in MLP vs {2} in DBN, cancel operation",
                                        cc,
                                        this.Layers[cc].NonBiasNeuronCount,
                                        layerSizes[cc]));
                                return;
                            }

                            if (layerSizes[cc] != this.Layers[mlpLayersCount - cc - 1].NonBiasNeuronCount)
                            {
                                Console.WriteLine(
                                    string.Format(
                                        "Layer {0} has different neuron count: {1} in MLP vs {2} in DBN, cancel operation",
                                        (mlpLayersCount - cc - 1),
                                        this.Layers[mlpLayersCount - cc - 1].NonBiasNeuronCount,
                                        layerSizes[cc]));
                                return;
                            }
                        }

                        Console.WriteLine("dbn.info checks ok...");
                    }
                    else
                    {
                        Console.WriteLine("dbn contains more layer than can be gets by this autoencoder, cancel operation");
                    }
                }
                else
                {
                    Console.WriteLine("dbn.info can't be parsed, check skipped...");
                }
            }
            else
            {
                Console.WriteLine("dbn.info not found, check skipped...");
            }

            #endregion


            Console.WriteLine("Load weights from DBN...");

            for (var layerIndex = 1; layerIndex <= Math.Min(this.Layers.Length, dbnFolderCount); layerIndex++)
            {
                var layer = Path.Combine(root, "rbm_layer" + (layerIndex - 1));
                var lastEpoche =
                    (from d in Directory.EnumerateDirectories(layer)
                     orderby this.ExtractRBMEpocheNumberFromDirName(d) descending
                     select d).First();

                Console.WriteLine(
                    string.Format(
                        "Processed layers {0}, {1} weights from: {2}",
                        layerIndex,
                        (mlpLayersCount - layerIndex),
                        lastEpoche));

                var pathToWeightsFile = Path.Combine(lastEpoche, "weights.bin");
                
                this.Layers[layerIndex].LoadWeightsFromRBM(pathToWeightsFile);
                this.Layers[mlpLayersCount - layerIndex].LoadAutoencoderWeightsFromRBM(pathToWeightsFile);
            }

            Console.WriteLine("Load weights done");
        }

        private int ExtractRBMEpocheNumberFromDirName(string dirname)
        {
            var lastSlashIndex = dirname.ToList().FindLastIndex(j => j == '\\');

            var stringNumber = dirname.Substring(lastSlashIndex + 8);

            return int.Parse(stringNumber);
        }

        public string DumpLayerInformation()
        {
            return
                string.Join("-", this.Layers.ToList().ConvertAll(j => j.DumpLayerInformation()));
        }

    }
}
