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
using MyNN.MLP2.Structure.Neurons.Function;
using MyNN.OutputConsole;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.Autoencoders
{
    public class StackedAutoencoder
    {
        private readonly IRandomizer _randomizer;
        private readonly ISerializationHelper _serialization;
        private readonly Func<DataSet, ITrainDataProvider> _dataProviderFactory;
        private readonly Func<DataSet, IValidation> _validationFactory;
        private readonly Func<int, ILearningAlgorithmConfig> _configFactory;
        private readonly IBackpropagationAlgorithmFactory _backpropagationAlgorithmFactory;
        private readonly IForwardPropagationFactory _forwardPropagationFactory;
        private readonly LayerInfo[] _layerInfos;

        public MLP CombinedNet
        {
            get;
            private set;
        }

        public StackedAutoencoder(
            IRandomizer randomizer,
            ISerializationHelper serialization,
            Func<DataSet, ITrainDataProvider> dataProviderFactory,
            Func<DataSet, IValidation> validationFactory,
            Func<int, ILearningAlgorithmConfig> configFactory,
            IBackpropagationAlgorithmFactory backpropagationAlgorithmFactory,
            IForwardPropagationFactory forwardPropagationFactory,
            params LayerInfo[] layerInfos)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
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
                throw new ArgumentException("В StackedAutoencoder надо задавать информации о слоях из первой половины автоенкодера, в отличии от Autoencoder");
            }

            _randomizer = randomizer;
            _serialization = serialization;
            _dataProviderFactory = dataProviderFactory;
            _validationFactory = validationFactory;
            _configFactory = configFactory;
            _backpropagationAlgorithmFactory = backpropagationAlgorithmFactory;
            _forwardPropagationFactory = forwardPropagationFactory;
            _layerInfos = layerInfos;
        }

        public MLP Train(
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
                throw new InvalidOperationException("trainData.IsAutoencoderDataSet: Не надо автоенкодер-датасеты, сами сделаем");
            }
            if (validationData.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("validationData.IsAutoencoderDataSet: Не надо автоенкодер-датасеты, сами сделаем");
            }

            var processingTrainData = trainData;
            var processingValidationData = validationData;

            //итоговый автоенкодер
            var layerList = new MLPLayer[_layerInfos.Length * 2 - 1];
            var layerListCount = layerList.Length;

            var depth = _layerInfos.Length - 1;
            for (var depthIndex = 0; depthIndex < depth; depthIndex++)
            {
                var net = new MLP(
                    _randomizer,
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
                        _layerInfos[depthIndex].LayerSize
                    });
                
                ConsoleAmbientContext.Console.WriteLine("Autoencoder created with conf: " + net.DumpLayerInformation());

                var trainDataProvider = _dataProviderFactory(processingTrainData);
                var validation = _validationFactory(processingValidationData);

                var config = _configFactory(depthIndex);

                this.Train3LayerAutoencoder(
                    net,
                    config,
                    validation,
                    trainDataProvider);

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

                if (depthIndex < depth - 1)
                {
                    //обновляем обучающие данные (от исходного множества, чтобы без применения возможных деформаций)
                    net.AutoencoderCutTail();

                    using (var clProvider = new CLProvider())
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

            //приделываем биас-нейроны
            for (var cc = (layerListCount + 1) / 2; cc < layerListCount - 1; cc++)
            {
                layerList[cc] = new MLPLayer(
                    layerList[cc].Neurons,
                    true);
            }

            //собираем итоговый автоенкодер
            this.CombinedNet = new MLP(
                _randomizer,
                root,
                null,
                layerList);

            using (var clProvider = new CLProvider())
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
                Path.Combine(root, this.CombinedNet.FolderName.ToLower() + ".sdae"));

            return
                this.CombinedNet;
        }

        private void Train3LayerAutoencoder(
            MLP net,
            ILearningAlgorithmConfig config,
            IValidation validation,
            ITrainDataProvider trainDataProvider)
        {
            using (var clProvider = new CLProvider())
            {
                var algo = _backpropagationAlgorithmFactory.GetBackpropagationAlgorithm(
                    _randomizer,
                    clProvider,
                    net,
                    validation,
                    config);

                algo.Train(trainDataProvider.GetDeformationDataSet);

            }

            //var takeIntoAccount = (int)(_partTakeOfAccount * net.Layers[1].NonBiasNeuronCount);

            //using (var clProvider = new CLProvider())
            //{

            //    var algo = new BackpropagationAlgorithm(
            //        _randomizer,
            //        (currentMLP, currentConfig) =>
            //            new CPUAutoencoderNLNCABackpropagationAlgorithm(
            //                VectorizationSizeEnum.VectorizationMode16,
            //                currentMLP,
            //                currentConfig,
            //                clProvider,
            //                (uzkii) => new DodfCalculatorOpenCL(
            //                    uzkii,
            //                    new VectorizedCpuDistanceDictCalculator()),
            //                1,
            //                _lambda,
            //                takeIntoAccount),
            //        net,
            //        validation,
            //        config);

            //    algo.Train(trainDataProvider.GetDeformationDataSet);

            //}

        }


    }
}
