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
using MyNN.MLP2.Structure.Neurons;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;

namespace MyNN.MLP2.Structure
{
    [Serializable]
    public class MLP
    {
        private readonly IRandomizer _randomizer;

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

        public MLPLayer[] Layers
        {
            get;
            private set;
        }


        public MLP(
            IRandomizer randomizer,
            string root,
            string folderName,
            IFunction[] activationFunction,
            params int[] neuronCountList)
        {
            //root, folderName  allowed to be null

            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (activationFunction == null || activationFunction.Length != neuronCountList.Length)
            {
                throw new InvalidOperationException("activationFunction == null || activationFunction.Length != neuronCountList.Length");
            }

            _randomizer = randomizer;

            Root = root;
            FolderName = folderName;

            //формируем слои
            this.Layers = new MLPLayer[neuronCountList.Length];

            //создаем входной слой
            this.Layers[0] = new MLPLayer(
                neuronCountList[0],
                true);

            //создаем скрытые слои и выходной слой
            var isPreviousLayerHadBiasNeuron = true;
            for (var cc = 1; cc < neuronCountList.Length; cc++)
            {
                var isLayerHasBiasNeuron = cc != (neuronCountList.Length - 1);

                this.Layers[cc] = new MLPLayer(
                    activationFunction[cc],
                    neuronCountList[cc],
                    neuronCountList[cc - 1],
                    isLayerHasBiasNeuron,
                    isPreviousLayerHadBiasNeuron,
                    _randomizer);

                isPreviousLayerHadBiasNeuron = isLayerHasBiasNeuron;
            }

            CreateWorkFolderFolder();
        }

        public MLP(
            IRandomizer randomizer,
            string root, 
            string folderName,
            MLPLayer[] layerList)
        {
            //root, folderName allowed to be null

            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (layerList == null)
            {
                throw new ArgumentNullException("layerList");
            }

            _randomizer = randomizer;

            Root = root;
            FolderName = folderName;
            
            //формируем слои
            this.Layers = layerList;
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

        /// <summary>
        /// Обрезать автоенкодер. Удаляются слои, начиная с узкого слоя и до конца
        /// </summary>
        public void AutoencoderCutTail()
        {
            var lls = new MLPLayer[(this.Layers.Length + 1) / 2];
            Array.Copy(this.Layers, 0, lls, 0, lls.Length);

            this.Layers = lls;

            //у последнего слоя убираем Bias нейрон
            var nll = this.Layers.Last();
            nll.RemoveBiasNeuron();
        }

        /// <summary>
        /// Убрать последний слой
        /// </summary>
        public void CutLastLayer()
        {
            var lls = new MLPLayer[this.Layers.Length - 1];
            Array.Copy(this.Layers, 0, lls, 0, lls.Length);

            this.Layers = lls;
        }


        /// <summary>
        /// Обрезать автоенкодер. Удаляются слои, начиная с первого и до узкого слоя
        /// </summary>
        public void AutoencoderCutHead()
        {
            var lls = new MLPLayer[(this.Layers.Length + 1) / 2];
            Array.Copy(this.Layers, this.Layers.Length - lls.Length, lls, 0, lls.Length);

            this.Layers = lls;
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

            var lastl = this.Layers[this.Layers.Length - 1];
            lastl.AddBiasNeuron();

            var newl = new MLPLayer[this.Layers.Length + 1];
            this.Layers.CopyTo(newl, 0);

            newl[this.Layers.Length] = new MLPLayer(
                activationFunction,
                nonBiasNeuronCount,
                lastl.NonBiasNeuronCount,
                isNeedBiasNeuron,
                true,
                _randomizer);

            this.Layers = newl;
        }

        public Bitmap GetVisualScheme()
        {
            const int imageWidth = 2000;
            const int imageHeight = 2000;
            const int neuronDiameter = 50;
            const int neuronRadius = 25;

            var result = new Bitmap(imageWidth, imageHeight);
            using (var g = Graphics.FromImage(result))
            {
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;

                g.Clear(Color.LightGreen);

                if (this.Layers.Length <= 5 && this.Layers.All(j => j.Neurons.Length < 50))
                {
                    // ------------------------- DRAW NEURONS -------------------------
                    using (var neuronFont = new Font("Times New Roman", 12, FontStyle.Regular))
                    {
                        for (var layerIndex = 0; layerIndex < this.Layers.Length; layerIndex++)
                        {
                            var l = this.Layers[layerIndex];

                            for (int neuronIndex = 0, neuronCount = l.Neurons.Length; neuronIndex < neuronCount; neuronIndex++)
                            {
                                var n = l.Neurons[neuronIndex];

                                using (var neuronColor = GetNeuronColor(layerIndex, neuronCount, neuronIndex, n))
                                {
                                    int x;
                                    int y;
                                    GetNeuronCenterPosition(
                                        imageWidth,
                                        imageHeight,
                                        this.Layers.Length,
                                        layerIndex,
                                        neuronCount,
                                        neuronIndex,
                                        neuronRadius,
                                        out x,
                                        out y
                                        );

                                    g.DrawEllipse(
                                        neuronColor,
                                        x - neuronRadius,
                                        y - neuronRadius,
                                        neuronDiameter,
                                        neuronDiameter);

                                    g.DrawString(
                                        GetNeuronActivationFunctionName(n), 
                                        neuronFont,
                                        neuronColor.Brush,
                                        x - (neuronDiameter*0.4f),
                                        y - (neuronDiameter*0.2f));
                                }
                            }
                        }
                    }

                    // ------------------------- DRAW WEIGHTS -------------------------
                    using (var weightFont = new Font("Times New Roman", 12, FontStyle.Regular))
                    {
                        for (var layerIndex = 0; layerIndex < this.Layers.Length; layerIndex++)
                        {
                            var l = this.Layers[layerIndex];

                            for (int neuronIndex = 0, neuronCount = l.Neurons.Length; neuronIndex < neuronCount; neuronIndex++)
                            {
                                var n = l.Neurons[neuronIndex];

                                for (int weightIndex = 0, weightCount = n.Weights.Length; weightIndex < weightCount; weightIndex++)
                                {
                                    using (var weightColor = GetWeightColor())
                                    {

                                        int xleft;
                                        int yleft;
                                        GetNeuronCenterPosition(
                                            imageWidth,
                                            imageHeight,
                                            this.Layers.Length,
                                            layerIndex - 1,
                                            this.Layers[layerIndex - 1].Neurons.Length,
                                            weightIndex,
                                            neuronRadius,
                                            out xleft,
                                            out yleft
                                            );

                                        int xright;
                                        int yright;
                                        GetNeuronCenterPosition(
                                            imageWidth,
                                            imageHeight,
                                            this.Layers.Length,
                                            layerIndex,
                                            neuronCount,
                                            neuronIndex,
                                            neuronRadius,
                                            out xright,
                                            out yright
                                            );

                                        var angle = Math.Atan2(yright - yleft, xright - xleft);

                                        var xleft2 = xleft + (int)(neuronRadius * Math.Cos(angle));
                                        var yleft2 = yleft + (int)(neuronRadius * Math.Sin(angle));

                                        var xright2 = xright - (int)(neuronRadius * Math.Cos(angle));
                                        var yright2 = yright - (int)(neuronRadius * Math.Sin(angle));

                                        g.DrawLine(
                                            weightColor,
                                            xleft2,
                                            yleft2,
                                            xright2,
                                            yright2);

                                        var xtext = xright - (int)((4 * neuronRadius) * Math.Cos(angle));
                                        var ytext = yright - (int)((4 * neuronRadius) * Math.Sin(angle));

                                        g.DrawString(
                                            n.Weights[weightIndex].ToString(),
                                            weightFont,
                                            weightColor.Brush,
                                            xtext - 40,
                                            ytext - 5);

                                    }
                                }
                            }
                        }
                    }
                }

            }

            return result;
        }

        private string GetNeuronActivationFunctionName(
            TrainableMLPNeuron neuron)
        {
            var result = string.Empty;

            if (neuron.IsBiasNeuron)
            {
                result = "Bias";
            }
            else
            {
                if (neuron.ActivationFunction != null)
                {
                    result = neuron.ActivationFunction.ShortName;
                }
                else
                {
                    result = "Input";
                }
            }

            return result;
        }

        private Pen GetWeightColor(
            )
        {
            Pen result = null;
            
            result = new Pen(Brushes.Black);

            return result;
        }

        private Pen GetNeuronColor(
            int layerIndex,
            int neuronCount,
            int neuronIndex,
            TrainableMLPNeuron neuron)
        {
            Pen result = null;

            if (neuronIndex == neuronCount - 1 && neuron.IsBiasNeuron)
            {
                result = new Pen(Brushes.Black);
            }
            else
            {
                result = layerIndex == 0 ? new Pen(Brushes.Red) : new Pen(Brushes.Green);
            }
            
            return result;
        }

        private void GetNeuronCenterPosition(
            int imageWidth,
            int imageHeight,
            int layerCount,
            int layerIndex,
            int neuronCount,
            int neuronIndex,
            int neuronRadius,
            out int x,
            out int y)
        {
            var wStep = imageWidth / layerCount;
            var leftShift = wStep / 2;

            var hStep = imageHeight / neuronCount;
            var topShift = hStep / 2;

            x = leftShift + wStep * layerIndex + neuronRadius ;
            y = topShift + hStep * neuronIndex + neuronRadius;
        }

        public string DumpLayerInformation()
        {
            return
                string.Join(" -> ", this.Layers.ToList().ConvertAll(j => j.DumpLayerInformation()));
        }

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
                .FindAll(j => Path.GetFileName(j).StartsWith("rbm_layer"))
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
                var layer = Path.Combine(root, "rbm_layer" + (layerIndex - 1));
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
                .FindAll(j => Path.GetFileName(j).StartsWith("rbm_layer"))
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
                var layer = Path.Combine(root, "rbm_layer" + (layerIndex - 1));
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

    }
}
