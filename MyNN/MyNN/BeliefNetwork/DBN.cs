using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.BeliefNetwork.Accuracy;
using MyNN.BeliefNetwork.ImageReconstructor;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.FreeEnergyCalculator;
using MyNN.BoltzmannMachines;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.Reconstructor;
using MyNN.Data;
using MyNN.LearningRateController;
using MyNN.MLP2.ArtifactContainer;
using MyNN.Randomizer;

namespace MyNN.BeliefNetwork
{
    public class DBN
    {
        private readonly IRandomizer _randomizer;
        private readonly IArtifactContainer _dbnContainer;
        private readonly int[] _layerSizes;

        public DBN(
            IRandomizer randomizer,
            IArtifactContainer dbnContainer,
            params int[] layerSizes
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (dbnContainer == null)
            {
                throw new ArgumentNullException("dbnContainer");
            }

            _randomizer = randomizer;
            _dbnContainer = dbnContainer;
            _layerSizes = layerSizes;
        }

        public void Train(
            IDataSet trainData,
            IDataSet validationData,
            IImageReconstructor imageReconstructor,
            ILearningRate learningRateController,
            IAccuracyController accuracyController,
            int batchSize,
            int maxGibbsChainLength
            )
        {
            var sir = new StackedImageReconstructor(imageReconstructor);

            var layerTrainData = trainData;
            var layerValidationData = validationData;

            for (var layerIndex = 0; layerIndex < _layerSizes.Length - 1; layerIndex++)
            {
                var visibleNeuronCount = _layerSizes[layerIndex];
                var hiddenNeuronCount = _layerSizes[layerIndex + 1];

                var rbmContainer = _dbnContainer.GetChildContainer(
                    string.Format(
                        "rbm{0}",
                        layerIndex));
                    
                IRBM rbm;
                IContainer container;
                IAlgorithm algorithm;
                this.CreateRBM(
                    _randomizer,
                    trainData,
                    rbmContainer,
                    sir,
                    visibleNeuronCount,
                    hiddenNeuronCount,
                    layerIndex,
                    out rbm,
                    out container,
                    out algorithm
                    );

                rbm.Train(
                    () =>
                    {
                        var binarized = trainData.Binarize(_randomizer);
                        return binarized;
                    },
                    validationData,
                    learningRateController,
                    accuracyController.Clone(),
                    batchSize,
                    maxGibbsChainLength);

                sir.AddConverter(
                    (d) =>
                    {
                        container.SetHidden(d);
                        var result = algorithm.CalculateVisible();
                        return result;
                    });

                layerTrainData = ForwardDataSet(
                    layerTrainData,
                    container,
                    algorithm);

                layerValidationData = ForwardDataSet(
                    layerValidationData, 
                    container, 
                    algorithm);

            }
        }

        private IDataSet ForwardDataSet(
            IDataSet dataSet, 
            IContainer container, 
            IAlgorithm algorithm)
        {
            if (dataSet == null)
            {
                throw new ArgumentNullException("dataSet");
            }
            if (container == null)
            {
                throw new ArgumentNullException("container");
            }
            if (algorithm == null)
            {
                throw new ArgumentNullException("algorithm");
            }

            var newdiList = new List<DataItem>();
            foreach (var di in dataSet)
            {
                container.SetInput(di.Input);
                var nextLayer = algorithm.CalculateVisible();

                var newdi = new DataItem(
                    nextLayer,
                    di.Output);

                newdiList.Add(newdi);
            }

            var result = new DataSet(newdiList);

            return result;
        }

        private void CreateRBM(
            IRandomizer randomizer,
            IDataSet trainData,
            IArtifactContainer rbmContainer,
            IImageReconstructor imageReconstructor,
            int visibleNeuronCount, 
            int hiddenNeuronCount,
            int rbmIndex,
            out IRBM rbm,
            out IContainer container,
            out IAlgorithm algorithm
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (trainData == null)
            {
                throw new ArgumentNullException("trainData");
            }
            if (rbmContainer == null)
            {
                throw new ArgumentNullException("rbmContainer");
            }
            if (imageReconstructor == null)
            {
                throw new ArgumentNullException("imageReconstructor");
            }

            var calculator = new BBCalculator(randomizer, visibleNeuronCount, hiddenNeuronCount);

            var feCalculator = new FloatArrayFreeEnergyCalculator(
                visibleNeuronCount,
                hiddenNeuronCount);

            var facontainer = new FloatArrayContainer(
                randomizer,
                feCalculator,
                visibleNeuronCount,
                hiddenNeuronCount);

            container = facontainer;

            algorithm = new CD(
                calculator,
                facontainer);

            var extractor =
                rbmIndex == 0
                    ? new IsolatedFeatureExtractor(
                        hiddenNeuronCount,
                        28,
                        28)
                    : null;

            rbm = new RBM(
                rbmContainer,
                randomizer,
                container,
                algorithm,
                imageReconstructor,
                extractor
                );

        }


    }
}
