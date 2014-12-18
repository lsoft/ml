using System;
using System.Collections.Generic;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.IterateHelper;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.DataSet.ItemLoader;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.Common.NewData.Item;
using MyNN.Common.NewData.Normalizer;
using MyNN.Common.Other;
using MyNN.Common.OutputConsole;
using MyNN.Common.Randomizer;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.BackpropagationFactory;
using MyNN.MLP.ForwardPropagationFactory;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.MLPContainer;
using MyNN.MLP.Structure;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.MLP.Autoencoders
{
    public class StackedAutoencoder : IStackedAutoencoder
    {
        private readonly IDeviceChooser _deviceChooser;
        private readonly IMLPContainerHelper _mlpContainerHelper;
        private readonly IRandomizer _randomizer;
        private readonly IDataSetFactory _dataSetFactory;
        private readonly IDataItemFactory _dataItemFactory;
        private readonly IMLPFactory _mlpFactory;
        private readonly Func<int, IDataSet, IDataSetProvider> _dataProviderFactory;
        private readonly Func<int, IDataSet, IArtifactContainer, IValidation> _validationFactory;
        private readonly Func<int, ILearningAlgorithmConfig> _configFactory;
        private readonly Func<CLProvider, IBackpropagationFactory> _backpropagationFactoryFunc;
        private readonly Func<CLProvider, IForwardPropagationFactory> _forwardPropagationFactoryFunc;
        private readonly LayerInfo[] _layerInfos;

        public StackedAutoencoder(
            IDeviceChooser deviceChooser,
            IMLPContainerHelper mlpContainerHelper,
            IRandomizer randomizer,
            IDataSetFactory dataSetFactory,
            IDataItemFactory dataItemFactory,
            IMLPFactory mlpFactory,
            Func<int, IDataSet, IDataSetProvider> dataProviderFactory,
            Func<int, IDataSet, IArtifactContainer, IValidation> validationFactory,
            Func<int, ILearningAlgorithmConfig> configFactory,
            Func<CLProvider, IBackpropagationFactory> backpropagationFactoryFunc,
            Func<CLProvider, IForwardPropagationFactory> forwardPropagationFactoryFunc,
            params LayerInfo[] layerInfos)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }
            if (mlpContainerHelper == null)
            {
                throw new ArgumentNullException("mlpContainerHelper");
            }
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (dataSetFactory == null)
            {
                throw new ArgumentNullException("dataSetFactory");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }
            if (mlpFactory == null)
            {
                throw new ArgumentNullException("mlpFactory");
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
            if (forwardPropagationFactoryFunc == null)
            {
                throw new ArgumentNullException("forwardPropagationFactoryFunc");
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

            _deviceChooser = deviceChooser;
            _mlpContainerHelper = mlpContainerHelper;
            _randomizer = randomizer;
            _dataSetFactory = dataSetFactory;
            _dataItemFactory = dataItemFactory;
            _mlpFactory = mlpFactory;
            _dataProviderFactory = dataProviderFactory;
            _validationFactory = validationFactory;
            _configFactory = configFactory;
            _backpropagationFactoryFunc = backpropagationFactoryFunc;
            _forwardPropagationFactoryFunc = forwardPropagationFactoryFunc;
            _layerInfos = layerInfos;
        }

        public IMLP Train(
            string sdaeName,
            IArtifactContainer rootContainer,
            IDataSet trainData,
            IDataSet validationData)
        {
            if (sdaeName == null)
            {
                throw new ArgumentNullException("sdaeName");
            }
            if (rootContainer == null)
            {
                throw new ArgumentNullException("rootContainer");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (!trainData.IsAutoencoderDataSet)
            {
                throw new InvalidOperationException("trainData.IsAutoencoderDataSet: Надо автоенкодер-датасеты.");
            }
            if (!validationData.IsAutoencoderDataSet)
            {
                throw new InvalidOperationException("validationData.IsAutoencoderDataSet: Надо автоенкодер-датасеты.");
            }

            var sdaeContainer = rootContainer.GetChildContainer(sdaeName);

            var processingTrainData = trainData;
            var processingValidationData = validationData;

            //итоговый автоенкодер
            var layerList = new ILayer[_layerInfos.Length*2 - 1];
            var layerListCount = layerList.Length;

            var depth = _layerInfos.Length - 1;
            for (var depthIndex = 0; depthIndex < depth; depthIndex++)
            {
                var mlpName = string.Format(
                    "mlp{0}.mlp",
                    DateTime.Now.ToString("yyyyMMddHHmmss"));

                var mlp = _mlpFactory.CreateMLP(
                    mlpName,
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

                ConsoleAmbientContext.Console.WriteLine("Autoencoder created with conf: " + mlp.GetLayerInformation());

                var mlpContainer = sdaeContainer.GetChildContainer(mlpName);

                var trainDataProvider = _dataProviderFactory(
                    depthIndex,
                    processingTrainData);
                var validation = _validationFactory(
                    depthIndex,
                    processingValidationData,
                    mlpContainer);

                var config = _configFactory(depthIndex);

                //обучаем автоенкодер
                using (var clProvider = new CLProvider(_deviceChooser, true))
                {
                    var backpropagationFactory = _backpropagationFactoryFunc(
                        clProvider
                        );

                    var algo = backpropagationFactory.CreateBackpropagation(
                        _randomizer,
                        clProvider,
                        mlpContainer,
                        mlp,
                        validation,
                        config);

                    algo.Train(trainDataProvider);
                }

                //добавляем в итоговый автоенкодер слои
                var cloned = sdaeContainer.DeepClone(mlp);
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
                    mlp.AutoencoderCutTail();

                    using (var clProvider = new CLProvider(_deviceChooser, true))
                    {
                        var forwardPropagationFactory = _forwardPropagationFactoryFunc(clProvider);

                        var forward = forwardPropagationFactory.Create(
                            _randomizer,
                            mlp);

                        var nextTrain = forward.ComputeOutput(processingTrainData);

                        var tdiList = new List<IDataItem>();
                        foreach (var nt in nextTrain)
                        {
                            var di = _dataItemFactory.CreateDataItem(
                                nt.NState,
                                nt.NState
                                );

                            tdiList.Add(di);
                        }

                        processingTrainData = _dataSetFactory.CreateDataSet(
                            new FromArrayDataItemLoader(
                                tdiList,
                                new DefaultNormalizer()),
                            0
                            );

                        //обновляем валидационные данные (от исходного множества, чтобы без применения возможных деформаций)
                        var nextValidation = forward.ComputeOutput(processingValidationData);

                        var vdiList = new List<IDataItem>();
                        foreach (var nv in nextValidation)
                        {
                            var di = _dataItemFactory.CreateDataItem(
                                nv.NState,
                                nv.NState
                                );

                            vdiList.Add(di);
                        }
                        processingValidationData = _dataSetFactory.CreateDataSet(
                            new FromArrayDataItemLoader(
                                vdiList,
                                new DefaultNormalizer()),
                            0
                            );
                    }
                }
            }

            //приделываем биас-нейроны
            for (var cc = (layerListCount + 1)/2; cc < layerListCount - 1; cc++)
            {
                layerList[cc].AddBiasNeuron();
            }

            //собираем итоговый автоенкодер
            var combinedNet = _mlpFactory.CreateMLP(
                sdaeName,
                layerList);

            using (var clProvider = new CLProvider(_deviceChooser, true))
            {
                var forwardPropagationFactory = _forwardPropagationFactoryFunc(clProvider);

                var finalForward = forwardPropagationFactory.Create(
                    _randomizer,
                    combinedNet);

                //валидируем его
                var finalValidation = _validationFactory(
                    0,
                    validationData,
                    sdaeContainer);

                var finalAccuracy = finalValidation.Validate(
                    finalForward,
                    0,
                    sdaeContainer
                    );

                //сохраняем
                _mlpContainerHelper.SaveMLP(
                    sdaeContainer,
                    combinedNet,
                    finalAccuracy
                    );
            }

            return
                combinedNet;
        }
    }
}
