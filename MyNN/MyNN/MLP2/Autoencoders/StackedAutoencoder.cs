using System;
using System.IO;
using System.Linq;
using MyNN.Data;
using MyNN.Data.TrainDataProvider;
using MyNN.MLP2.ArtifactContainer;
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
    public class StackedAutoencoder : IStackedAutoencoder
    {
        private readonly IDeviceChooser _deviceChooser;
        private readonly IRandomizer _randomizer;
        private readonly IMLPFactory _mlpFactory;
        private readonly Func<IDataSet, ITrainDataProvider> _dataProviderFactory;
        private readonly Func<IDataSet, IArtifactContainer, IValidation> _validationFactory;
        private readonly Func<int, ILearningAlgorithmConfig> _configFactory;
        private readonly IBackpropagationAlgorithmFactory _backpropagationAlgorithmFactory;
        private readonly IForwardPropagationFactory _forwardPropagationFactory;
        private readonly LayerInfo[] _layerInfos;

        public StackedAutoencoder(
            IDeviceChooser deviceChooser,
            IRandomizer randomizer,
            IMLPFactory mlpFactory,
            Func<IDataSet, ITrainDataProvider> dataProviderFactory,
            Func<IDataSet, IArtifactContainer, IValidation> validationFactory,
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
            _randomizer = randomizer;
            _mlpFactory = mlpFactory;
            _dataProviderFactory = dataProviderFactory;
            _validationFactory = validationFactory;
            _configFactory = configFactory;
            _backpropagationAlgorithmFactory = backpropagationAlgorithmFactory;
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
            var layerList = new ILayer[_layerInfos.Length * 2 - 1];
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

                var trainDataProvider = _dataProviderFactory(processingTrainData);
                var validation = _validationFactory(
                    processingValidationData,
                    mlpContainer);

                var config = _configFactory(depthIndex);

                //обучаем автоенкодер
                using (var clProvider = new CLProvider())
                {
                    var algo = _backpropagationAlgorithmFactory.GetBackpropagationAlgorithm(
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
                        var forward = _forwardPropagationFactory.Create(_randomizer, clProvider, mlp);

                        var nextTrain = forward.ComputeOutput(processingTrainData);
                        var newTrainData = new DataSet(trainData, nextTrain.ConvertAll(j => j.State));
                        processingTrainData = newTrainData;

                        //обновляем валидационные данные (от исходного множества, чтобы без применения возможных деформаций)
                        var nextValidation = forward.ComputeOutput(processingValidationData);
                        var newValidationData = new DataSet(validationData, nextValidation.ConvertAll(j => j.State));
                        processingValidationData = newValidationData;
                    }
                }
            }

            //приделываем биас-нейроны
            for (var cc = (layerListCount + 1) / 2; cc < layerListCount - 1; cc++)
            {
                layerList[cc].AddBiasNeuron();
            }

            //собираем итоговый автоенкодер
            var combinedNet = _mlpFactory.CreateMLP(
                sdaeName,
                layerList);

            using (var clProvider = new CLProvider())
            {
                var forward = _forwardPropagationFactory.Create(_randomizer, clProvider, combinedNet);

                //валидируем его
                var finalValidation = _validationFactory(
                    validationData,
                    sdaeContainer);

                var finalAccuracy =finalValidation.Validate(
                    forward,
                    0,
                    sdaeContainer
                    );

                //сохраняем
                sdaeContainer.SaveMLP(
                    combinedNet, 
                    finalAccuracy
                    );
            }

            return
                combinedNet;
        }


    }
}
