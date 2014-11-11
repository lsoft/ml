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
            long arraySize
            )
        {
            if (arraySize <= 0)
            {
                throw new ArgumentOutOfRangeException("arraySize");
            }

            return 
                new BigArrayWeightMaskContainer(
                    _clProvider,
                    arraySize,
                    _randomizer
                    );
        }
    }
}