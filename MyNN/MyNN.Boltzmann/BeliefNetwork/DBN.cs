using System;
using System.Linq;
using System.Text;
using MyNN.Boltzmann.BeliefNetwork.Accuracy;
using MyNN.Boltzmann.BeliefNetwork.ImageReconstructor;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.DataSetConverter;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.LearningRateController;

namespace MyNN.Boltzmann.BeliefNetwork
{
    public class DBN
    {
        public const string RbmFolderName = "rbm_layer";

        private readonly IArtifactContainer _dbnContainer;
        private readonly IRBMFactory _rbmFactory;
        private readonly int[] _layerSizes;

        public DBN(
            IArtifactContainer dbnContainer,
            IRBMFactory rbmFactory,
            params int[] layerSizes
            )
        {
            if (dbnContainer == null)
            {
                throw new ArgumentNullException("dbnContainer");
            }
            if (rbmFactory == null)
            {
                throw new ArgumentNullException("rbmFactory");
            }

            _dbnContainer = dbnContainer;
            _rbmFactory = rbmFactory;
            _layerSizes = layerSizes;
        }

        public void Train(
            IDataSet trainData,
            IDataSet validationData,
            Func<IDataSet, ITrainDataProvider> trainDataProviderFunc,
            IFeatureExtractorFactory featureExtractorFactory,
            IStackedImageReconstructor stackedImageReconstructor,
            ILearningRate learningRateController,
            IAccuracyController accuracyController,
            int batchSize,
            int maxGibbsChainLength
            )
        {
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (validationData == null)
            {
                throw new ArgumentNullException("validationData");
            }
            if (trainDataProviderFunc == null)
            {
                throw new ArgumentNullException("trainDataProviderFunc");
            }
            if (featureExtractorFactory == null)
            {
                throw new ArgumentNullException("featureExtractorFactory");
            }
            if (stackedImageReconstructor == null)
            {
                throw new ArgumentNullException("stackedImageReconstructor");
            }
            if (learningRateController == null)
            {
                throw new ArgumentNullException("learningRateController");
            }
            if (accuracyController == null)
            {
                throw new ArgumentNullException("accuracyController");
            }

            var layerTrainData = trainData;
            var layerValidationData = validationData;

            for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
            {
                var visibleNeuronCount = _layerSizes[layerIndex];
                var hiddenNeuronCount = _layerSizes[layerIndex + 1];

                var rbmContainer = _dbnContainer.GetChildContainer(RbmFolderName + layerIndex);

                var featureExractor =
                    layerIndex == 0
                        ? featureExtractorFactory.CreateFeatureExtractor(hiddenNeuronCount)
                        : null;

                IRBM rbm;
                IDataSetConverter forwardDataSetConverter;
                IDataArrayConverter dataArrayConverter;
                _rbmFactory.CreateRBM(
                    layerTrainData,
                    rbmContainer,
                    stackedImageReconstructor,
                    featureExractor,
                    visibleNeuronCount,
                    hiddenNeuronCount,
                    out rbm,
                    out forwardDataSetConverter,
                    out dataArrayConverter
                    );

                var trainDataProvider = trainDataProviderFunc(layerTrainData);

                rbm.Train(
                    trainDataProvider,
                    layerValidationData,
                    learningRateController,
                    accuracyController.Clone(),
                    batchSize,
                    maxGibbsChainLength);

                stackedImageReconstructor.AddConverter(dataArrayConverter);

                layerTrainData = forwardDataSetConverter.Convert(layerTrainData);
                layerValidationData = forwardDataSetConverter.Convert(layerValidationData);

            }

            this.DumpDBNInfo();
        }

        private void DumpDBNInfo()
        {
            var dbninfo = string.Format(
                "Layers sizes: {0}",
                string.Join("-", _layerSizes.ToList().ConvertAll(k => k.ToString()))
                );

            using (var s = _dbnContainer.GetWriteStreamForResource("dbn.info"))
            {
                var bytes = Encoding.ASCII.GetBytes(dbninfo);
                s.Write(bytes, 0, bytes.Length);
            }
        }

    }
}
