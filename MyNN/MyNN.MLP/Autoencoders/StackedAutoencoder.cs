using System;
using System.Linq;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.Data.TrainDataProvider;
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
        private readonly IDataItemFactory _dataItemFactory;
        private readonly IMLPFactory _mlpFactory;
        private readonly Func<int, IDataSet, ITrainDataProvider> _dataProviderFactory;
        private readonly Func<int, IDataSet, IArtifactContainer, IValidation> _validationFactory;
        private readonly Func<int, ILearningAlgorithmConfig> _configFactory;
        private readonly IBackpropagationFactory _backpropagationFactory;
        private readonly IForwardPropagationFactory _forwardPropagationFactory;
        private readonly LayerInfo[] _layerInfos;

        public StackedAutoencoder(
            IDeviceChooser deviceChooser,
            IMLPContainerHelper mlpContainerHelper,
            IRandomizer randomizer,
            IDataItemFactory dataItemFactory,
            IMLPFactory mlpFactory,
            Func<int, IDataSet, ITrainDataProvider> dataProviderFactory,
            Func<int, IDataSet, IArtifactContainer, IValidation> validationFactory,
            Func<int, ILearningAlgorithmConfig> configFactory,
            IBackpropagationFactory backpropagationFactory,
            IForwardPropagationFactory forwardPropagationFactory,
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

            _deviceChooser = deviceChooser;
            _mlpContainerHelper = mlpContainerHelper;
            _randomizer = randomizer;
            _dataItemFactory = dataItemFactory;
            _mlpFactory = mlpFactory;
            _dataProviderFactory = dataProviderFactory;
            _validationFactory = validationFactory;
            _configFactory = configFactory;
            _backpropagationFactory = backpropagationFactory;
            _forwardPropagationFactory = forwardPropagationFactory;
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
            if (trainData.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("trainData.IsAutoencoderDataSet: Не надо автоенкодер-датасеты, сами сделаем");
            }
            if (validationData.IsAuencoderDataSet)
            {
                throw new InvalidOperationException("validationData.IsAutoencoderDataSet: Не надо автоенкодер-датасеты, сами сделаем");
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
                using (var clProvider = new CLProvider())
                {
                    var algo = _backpropagationFactory.CreateBackpropagation(
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

                    var forward = _forwardPropagationFactory.Create(
                        _randomizer,
                        mlp);

                    var nextTrain = forward.ComputeOutput(processingTrainData);
                    var newTrainData = new DataSet(
                        trainData,
                        nextTrain.ConvertAll(j => j.NState),
                        _dataItemFactory);
                    processingTrainData = newTrainData;

                    //обновляем валидационные данные (от исходного множества, чтобы без применения возможных деформаций)
                    var nextValidation = forward.ComputeOutput(processingValidationData);
                    var newValidationData = new DataSet(
                        validationData,
                        nextValidation.ConvertAll(j => j.NState),
                        _dataItemFactory);
                    processingValidationData = newValidationData;
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
            
            var finalForward = _forwardPropagationFactory.Create(
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

            return
                combinedNet;
        }
    }
}
