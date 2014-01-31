using System;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.ForwardFactory
{
    public class OpenCLForwardPropagationFactory : IForwardPropagationFactory
    {
        private readonly VectorizationSizeEnum _vse;

        public OpenCLForwardPropagationFactory(
            VectorizationSizeEnum vse = VectorizationSizeEnum.VectorizationMode16)
        {
            _vse = vse;
        }

        public IForwardPropagation Create(
            IRandomizer randomizer,
            CLProvider clProvider,
            MLP mlp)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }
            return 
                new CPUForwardPropagation(
                    _vse,
                    mlp,
                    clProvider);
        }
    }
}