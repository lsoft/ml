using System;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Algorithm;
using MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Container;

namespace MyNN.Boltzmann.BeliefNetwork.ImageReconstructor.Converter
{
    public class ImageReconstructorDataConverter : IDataArrayConverter
    {
        private readonly IContainer _container;
        private readonly IAlgorithm _algorithm;

        public ImageReconstructorDataConverter(
            IContainer container,
            IAlgorithm algorithm)
        {
            if (container == null)
            {
                throw new ArgumentNullException("container");
            }
            if (algorithm == null)
            {
                throw new ArgumentNullException("algorithm");
            }

            _container = container;
            _algorithm = algorithm;
        }

        public float[] Convert(float[] dataToConvert)
        {
            if (dataToConvert == null)
            {
                throw new ArgumentNullException("dataToConvert");
            }

            _container.SetHidden(dataToConvert);
                
            var result = _algorithm.CalculateVisible();
                
            return result;
        }
    }
}