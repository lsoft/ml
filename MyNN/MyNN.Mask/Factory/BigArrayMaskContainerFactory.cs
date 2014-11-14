using System;
using MyNN.Common.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.Mask.Factory
{
    public class BigArrayMaskContainerFactory : IOpenCLMaskContainerFactory
    {
        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;

        public BigArrayMaskContainerFactory(
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

        public IOpenCLMaskContainer CreateContainer(
            long arraySize,
            float p
            )
        {
            if (arraySize <= 0)
            {
                throw new ArgumentOutOfRangeException("arraySize");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            return 
                new BigArrayMaskContainer(
                    _clProvider,
                    arraySize,
                    _randomizer,
                    p
                    );
        }
    }
}