using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.Data.TypicalDataProvider;
using MyNN.NeuralNet;
using MyNN.NeuralNet.Computers;
using MyNN.NeuralNet.LearningConfig;
using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons;
using MyNN.NeuralNet.Structure.Neurons.Function;
using MyNN.NeuralNet.Train;
using MyNN.NeuralNet.Train.Algo;
using MyNN.NeuralNet.Train.Validation;

namespace MyNN.Autoencoders
{
    public class StackedAutoencoder
    {
        private readonly Func<DataSet, ITrainDataProvider> _dataProviderFactory;
        private readonly Func<DataSet, IValidation> _validationFactory;
        private readonly Func<int, ILearningAlgorithmConfig> _configFactory;
        private readonly LayerInfo[] _layerInfos;

        public MultiLayerNeuralNetwork CombinedNet
        {
            get;
            private set;
        }

        public StackedAutoencoder(
            Func<DataSet, ITrainDataProvider> dataProviderFactory,
            Func<DataSet, IValidation> validationFactory,
            Func<int, ILearningAlgorithmConfig> configFactory,
            params LayerInfo[] layerInfos)
        {
            if (dataProviderFactory == null)
            {
                throw new ArgumentNullException("dataProviderFactory");
            }
            if (validationFactory == null)
            {
                throw new ArgumentNullException("validationFactory");
            }
            if (configFactory == null)
            {
                throw new ArgumentNullException("configFactory");
            }
            if (layerInfos == null)
            {
                throw new ArgumentNullException("layerInfos");
            }
            if (layerInfos.Length < 2)
            {
                throw new ArgumentException("layerInfos");
            }
            if (layerInfos.First().LayerSize == layerInfos.Last().LayerSize)
            {
                throw new ArgumentException("В StackedAutoencoder надо задавать информации о слоях из первой половины автоенкодера, в отличии от Autoencoder");
            }

            _dataProviderFactory = dataProviderFactory;
            _validationFactory = validationFactory;
            _configFactory = configFactory;
            _layerInfos = layerInfos;
        }

        public MultiLayerNeuralNetwork Train(
            ref int rndSeed, 
            string root,
            DataSet trainData,
            DataSet validationData)
        {
            if (root == null)
            {
                throw new ArgumentNullException("root");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (trainData.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("trainData.IsAuencoderDataSet: Не надо автоенкодер-датасеты, сами сделаем");
            }
            if (validationData.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("validationData.IsAuencoderDataSet: Не надо автоенкодер-датасеты, сами сделаем");
            }

            var processingTrainData = trainData;
            var processingValidationData = validationData;

            //итоговый автоенкодер
            var layerList = new Layer[_layerInfos.Length * 2 - 1];
            var layerListCount = layerList.Length;

            var depth = _layerInfos.Length - 1;
            for (var depthIndex = 0; depthIndex < depth; depthIndex++)
            {
                var net = new MultiLayerNeuralNetwork(
                    root,
                    null,
                    new IFunction[3]
                    {
                        null,
                        _layerInfos[depthIndex + 1].ActivationFunction,
                        _layerInfos[depthIndex].ActivationFunction
                    },
                    ref rndSeed,
                    new int[3]
                    {
                        _layerInfos[depthIndex].LayerSize,
                        _layerInfos[depthIndex + 1].LayerSize,
                        _layerInfos[depthIndex].LayerSize
                    });

                Console.WriteLine("Network does not found. Created with conf: " + net.DumpLayerInformation());

                var trainDataProvider = _dataProviderFactory(processingTrainData);
                var validation = _validationFactory(processingValidationData);

                using (var universe = new VNNCLProvider(net))
                {
                    //создаем объект просчитывающий сеть
                    var computer =
                        new VOpenCLComputer(universe, true);

                    net.SetComputer(computer);

                    var config = _configFactory(depthIndex);

                    var alg =
                        //new NaiveBackpropagationLearningAlgorithm(
                        //new OpenCLNaiveBackpropAlgorithm(
                        new VOpenCLBackpropAlgorithm(
                            net,
                            config,
                            validation.Validate,
                            universe,
                            rndSeed);

                    //обучение сети
                    alg.Train(trainDataProvider.GetDeformationDataSet);

                    //добавляем в итоговый автоенкодер слои
                    if (depthIndex == 0)
                    {
                        layerList[0] = net.Layers[0];
                        layerList[1] = net.Layers[1];
                        layerList[layerListCount - 1] = net.Layers[2];
                    }
                    else
                    {
                        layerList[depthIndex + 1] = net.Layers[1];
                        layerList[layerListCount - 1 - depthIndex] = net.Layers[2];
                    }

                }

                if (depthIndex < depth - 1)
                {
                    //обновляем обучающие данные (от исходного множества, чтобы без применения возможных деформаций)
                    net.Layers = new Layer[]
                    {
                        net.Layers[0],
                        net.Layers[1]
                    };

                    using (var universe = new VNNCLProvider(net))
                    {
                        //создаем объект просчитывающий сеть
                        var computer =
                            new VOpenCLComputer(universe, true);

                        net.SetComputer(computer);


                        var nextTrain = net.ComputeOutput(processingTrainData.GetInputPart());
                        //отламываем биас
                        //for (var cc = 0; cc < nextTrain.Count; cc++)
                        //{
                        //    var b = new float[nextTrain[cc].Length - 1];
                        //    Array.Copy(nextTrain[cc], 0, b, 0, b.Length);
                        //    nextTrain[cc] = b;
                        //}
                        var newTrainData = new DataSet(trainData, nextTrain, null);
                        processingTrainData = newTrainData;

                        //обновляем валидационные данные (от исходного множества, чтобы без применения возможных деформаций)
                        var nextValidation = net.ComputeOutput(processingValidationData.GetInputPart());
                        //отламываем биас
                        //for (var cc = 0; cc < nextValidation.Count; cc++)
                        //{
                        //    var b = new float[nextValidation[cc].Length - 1];
                        //    Array.Copy(nextValidation[cc], 0, b, 0, b.Length);
                        //    nextValidation[cc] = b;
                        //}
                        var newValidationData = new DataSet(validationData, nextValidation, null);
                        processingValidationData = newValidationData;
                    }
                }
            }

            //приделываем биас-нейроны
            for (var cc = (layerListCount + 1) / 2; cc < layerListCount - 1; cc++)
            {
                layerList[cc] = new Layer(layerList[cc].Neurons, true);
            }

            //собираем итоговый автоенкодер
            this.CombinedNet = new MultiLayerNeuralNetwork(
                root,
                null,
                layerList);

            this.CombinedNet.SetComputer(new DefaultComputer(this.CombinedNet));

            //валидируем его
            var finalValidation = _validationFactory(validationData);
            
            finalValidation.Validate(
                this.CombinedNet,
                root,
                float.MaxValue,
                false);

            //сохраняем
            SerializationHelper.SaveToFile(
                this.CombinedNet,
                Path.Combine(root, this.CombinedNet.FolderName.ToLower() + ".mynn"));

            return
                this.CombinedNet;
        }

    }
}
