using System;
using MyNN.Common.Randomizer;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.WeightMask.Factory
{
    public class BigArrayWeightMaskContainerFactory : IOpenCLWeightMaskContainerFactory
    {
        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;

        public BigArrayWeightMaskContainerFactory(
            IRandomizer randomizer,
            CLProvider clProvider
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            _randomizer = randomizer;
            _clProvider = clProvider;
        }

        public IOpenCLWeightMaskContainer CreateContainer(
            ILayerConfiguration previousLayerConfiguration, 
            ILayerConfiguration currentLayerConfiguration)
        {
            if (previousLayerConfiguration == null)
            {
                throw new ArgumentNullException("previousLayerConfiguration");
            }
            if (currentLayerConfiguration == null)
            {
                throw new ArgumentNullException("currentLayerConfiguration");
            }

            return 
                new BigArrayWeightMaskContainer(
                    _clProvider,
                    previousLayerConfiguration,
                    currentLayerConfiguration,
                    _randomizer
                    );
        }
    }
}