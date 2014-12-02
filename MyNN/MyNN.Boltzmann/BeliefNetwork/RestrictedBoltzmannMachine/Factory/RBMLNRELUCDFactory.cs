using System;
using MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.Converter;
using MyNN.Boltzmann.BeliefNetwork.ImageReconstructor.Converter;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Calculator;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.Boltzmann.BoltzmannMachines;
using MyNN.Boltzmann.BoltzmannMachines.BinaryBinary.DBN.RBM.Feature;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Randomizer;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Factory
{
    public class RBMLNRELUCDFactory : IRBMFactory
    {
        private readonly IRandomizer _randomizer;
        private readonly IDataItemFactory _dataItemFactory;
        private readonly IDataSetFactory _dataSetFactory;

        public RBMLNRELUCDFactory(
            IRandomizer randomizer,
            IDataItemFactory dataItemFactory,
            IDataSetFactory dataSetFactory
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (dataItemFactory == null)
            {
                throw new ArgumentNullException("dataItemFactory");
            }
            if (dataSetFactory == null)
            {
                throw new ArgumentNullException("dataSetFactory");
            }

            _randomizer = randomizer;
            _dataItemFactory = dataItemFactory;
            _dataSetFactory = dataSetFactory;
        }

        public void CreateRBM(
            IDataSet trainData,
            IArtifactContainer rbmContainer,
            IImageReconstructor imageReconstructor,
            IFeatureExtractor featureExtractor,
            int visibleNeuronCount,
            int hiddenNeuronCount,
            out IRBM rbm,
            out IDataSetConverter forwardDataSetConverter,
            out IDataArrayConverter dataArrayConverter
            )
        {
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
            //featureExtractor allowed to be null

            var calculator = new LNRELUCalculator(
                visibleNeuronCount,
                hiddenNeuronCount);

            var container = new FloatArrayContainer(
                _randomizer,
                null,
                visibleNeuronCount,
                hiddenNeuronCount);

            var algorithm = new CD(
                calculator,
                container);

            rbm = new MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.RBM(
                rbmContainer,
                container,
                algorithm,
                imageReconstructor,
                featureExtractor
                );

            forwardDataSetConverter = new DBNDataSetConverter(
                container,
                algorithm,
                _dataItemFactory,
                _dataSetFactory
                );

            dataArrayConverter = new ImageReconstructorDataConverter(
                container,
                algorithm);
        }
    }
}