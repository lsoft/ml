using System;
using MyNN.MLP2.ForwardPropagation.DropConnect;
using MyNN.MLP2.OpenCL;
using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.ForwardFactory
{
    public class CPUDropConnectForwardPropagationFactory<T> : IForwardPropagationFactory
        where T : ILayerInference
    {
        private readonly VectorizationSizeEnum _vse;
        private readonly int _sampleCount;
        private readonly float _p;

        public CPUDropConnectForwardPropagationFactory(
            VectorizationSizeEnum vse = VectorizationSizeEnum.VectorizationMode16,
            int sampleCount = 10000,
            float p = 0.5f)
        {
            _vse = vse;
            _sampleCount = sampleCount;
            _p = p;
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
            var forwardPropagation = new InferenceOpenCLForwardPropagation<T>(
                _vse,
                mlp,
                clProvider,
                randomizer,
                _sampleCount,
                _p);

            return forwardPropagation;
        }
    }
}
