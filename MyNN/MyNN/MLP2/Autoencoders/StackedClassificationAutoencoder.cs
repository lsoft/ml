﻿using System;
using System.IO;
using System.Linq;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.BackpropagationFactory;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagationFactory;
using MyNN.MLP2.LearningConfig;

using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer;
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.MLP2.Autoencoders
{
    public class StackedClassificationAutoencoder : IStackedAutoencoder
    {
        private readonly IDeviceChooser _deviceChooser;
        private readonly IRandomizer _randomizer;
        private readonly IMLPFactory _mlpFactory;
        private readonly ISerializationHelper _serialization;
        private readonly Func<int, DataSet, ITrainDataProvider> _dataProviderFactory;
        private readonly Func<DataSet, IValidation> _validationFactory;
        private readonly Func<int, ILearningAlgorithmConfig> _configFactory;
        private readonly IBackpropagationAlgorithmFactory _backpropagationAlgorithmFactory;
        private readonly IForwardPropagationFactory _forwardPropagationFactory;
        private readonly LayerInfo[] _layerInfos;

        public IMLP CombinedNet
        {
            get;
            private set;
        }

        public StackedClassificationAutoencoder(
            IDeviceChooser deviceChooser,
            IRandomizer randomizer,
            IMLPFactory mlpFactory,
            ISerializationHelper serialization,
            Func<int, DataSet, ITrainDataProvider> dataProviderFactory,
            Func<DataSet, IValidation> validationFactory,
            Func<int, ILearningAlgorithmConfig> configFactory,
            IBackpropagationAlgorithmFactory backpropagationAlgorithmFactory,
            IForwardPropagationFactory forwardPropagationFactory,
            params LayerInfo[] layerInfos)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (mlpFactory == null)
            {
                throw new ArgumentNullException("mlpFactory");
            }
            if (serialization == null)
            {
                throw new ArgumentNullException("serialization");
            }
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
            if (backpropagationAlgorithmFactory == null)
            {
                throw new ArgumentNullException("backpropagationAlgorithmFactory");
            }
            if (forwardPropagationFactory == null)
            {
                throw new ArgumentNullException("forwardPropagationFactory");
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
                throw new ArgumentException("В StackedClassificationAutoencoder надо задавать информации о слоях из первой половины автоенкодера, в отличии от Autoencoder");
            }

            _deviceChooser = deviceChooser;
            _randomizer = randomizer;
            _mlpFactory = mlpFactory;
            _serialization = serialization;
            _dataProviderFactory = dataProviderFactory;
            _validationFactory = validationFactory;
            _configFactory = configFactory;
            _backpropagationAlgorithmFactory = backpropagationAlgorithmFactory;
            _forwardPropagationFactory = forwardPropagationFactory;
            _layerInfos = layerInfos;
        }

        public IMLP Train(
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
            if (trainData.IsClassificationAuencoderDataSet)
            {
                throw new InvalidOperationException("trainData.IsClassificationAuencoderDataSet: Не надо автоенкодер-датасеты, сами сделаем");
            }
            if (validationData.IsClassificationAuencoderDataSet)
            {
                throw new InvalidOperationException("validationData.IsClassificationAuencoderDataSet: Не надо автоенкодер-датасеты, сами сделаем");
            }

            var classificationLength = trainData[0].OutputLength;

            var processingTrainData = trainData;
            var processingValidationData = validationData;

            //итоговый автоенкодер
            var layerList = new ILayer[_layerInfos.Length * 2 - 1];
            var layerListCount = layerList.Length;

            var depth = _layerInfos.Length - 1;
            for (var depthIndex = 0; depthIndex < depth; depthIndex++)
            {
                var net = _mlpFactory.CreateMLP(
                    root,
                    null,
                    new IFunction[3]
                    {
                        null,
                        _layerInfos[depthIndex + 1].ActivationFunction,
                        _layerInfos[depthIndex].ActivationFunction
                    },
                    new int[3]
                    {
                        _layerInfos[depthIndex].LayerSize,
                        _layerInfos[depthIndex + 1].LayerSize,
                        _layerInfos[depthIndex].LayerSize + classificationLength
                    });

                ConsoleAmbientContext.Console.WriteLine("Autoencoder created with conf: " + net.GetLayerInformation());

                var trainDataProvider = _dataProviderFactory(depthIndex, processingTrainData);
                var validationDataProvider = _validationFactory(processingValidationData);

                using (var clProvider = new CLProvider(_deviceChooser, true))
                {
                    var config = _configFactory(depthIndex);

                    var algo = _backpropagationAlgorithmFactory.GetBackpropagationAlgorithm(
                        _randomizer,
                        clProvider,
                        net,
                        validationDataProvider,
                        config);

                    algo.Train(trainDataProvider);

                    //добавляем в итоговый автоенкодер слои
                    var cloned = _serialization.DeepClone(net);
                    if (depthIndex == 0)
                    {
                        layerList[0] = cloned.Layers[0];
                        layerList[1] = cloned.Layers[1];
                        layerList[layerListCount - 1] = cloned.Layers[2];
                    }
                    else
                    {
                        layerList[depthIndex + 1] = cloned.Layers[1];
                        layerList[layerListCount - 1 - depthIndex] = cloned.Layers[2];
                    }
                }

                if (depthIndex < depth - 1)
                {
                    //обновляем обучающие данные (от исходного множества, чтобы без применения возможных деформаций)
                    net.AutoencoderCutTail();

                    using (var clProvider = new CLProvider(_deviceChooser, true))
                    {
                        //var forward = new CPUForwardPropagation(
                        //    VectorizationSizeEnum.VectorizationMode16,
                        //    net,
                        //    clProvider);
                        var forward = _forwardPropagationFactory.Create(_randomizer, clProvider, net);

                        var nextTrain = forward.ComputeOutput(processingTrainData);
                        var newTrainData = new DataSet(trainData, nextTrain.ConvertAll(j => j.State), null);
                        processingTrainData = newTrainData;

                        //обновляем валидационные данные (от исходного множества, чтобы без применения возможных деформаций)
                        var nextValidation = forward.ComputeOutput(processingValidationData);
                        var newValidationData = new DataSet(validationData, nextValidation.ConvertAll(j => j.State), null);
                        processingValidationData = newValidationData;
                    }
                }
            }

            //удаляем нейроны, которые отвечали за классификацию (кроме последнего слоя, так как все равно должен получиться SCDAE)
            for (var cc = (layerListCount + 1) / 2; cc < layerListCount - 1; cc++)
            {
                throw new NotImplementedException("Эта операция должна делаться из MLPSurgeon!!! Переделать!");

                //layerList[cc] = new Layer(
                //    layerList[cc].Neurons.GetSubArray(classificationLength),
                //    false);
            }


            //приделываем биас-нейроны
            for (var cc = (layerListCount + 1) / 2; cc < layerListCount - 1; cc++)
            {
                layerList[cc].AddBiasNeuron();

                //layerList[cc] = new Layer(
                //    layerList[cc].Neurons,
                //    true);
            }

            //собираем итоговый автоенкодер
            this.CombinedNet = _mlpFactory.CreateMLP(
                root,
                null,
                layerList);

            using (var clProvider = new CLProvider(_deviceChooser, true))
            {
                //var forward = new CPUForwardPropagation(
                //    VectorizationSizeEnum.VectorizationMode16,
                //    this.CombinedNet,
                //    clProvider);
                var forward = _forwardPropagationFactory.Create(_randomizer, clProvider, this.CombinedNet);


                //валидируем его
                var finalValidation = _validationFactory(validationData);

                finalValidation.Validate(
                    forward,
                    root,
                    false);
            }

            //сохраняем
            _serialization.SaveToFile(
                this.CombinedNet,
                Path.Combine(root, this.CombinedNet.FolderName.ToLower() + ".scdae"));

            return
                this.CombinedNet;
        }

    }
}
