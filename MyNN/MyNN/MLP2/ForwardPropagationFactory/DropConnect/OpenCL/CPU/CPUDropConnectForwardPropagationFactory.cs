using System;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU;
using MyNN.MLP2.ForwardPropagation.DropConnect.Inference.OpenCL.CPU.Inferencer;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagationFactory.DropConnect.OpenCL.CPU
{
    /// <summary>
    /// Factory for forward propagator for dropconnect backpropagation algorithm than enables CPU-OpenCL
    /// For details about dropconnect please refer http://cs.nyu.edu/~wanli/dropc/
    /// </summary>
    /// <typeparam name="T">Dropconnect layer inferencer</typeparam>
    public class CPUDropConnectForwardPropagationFactory<T> : IForwardPropagationFactory
        where T : ILayerInference
    {
        private readonly VectorizationSizeEnum _vse;
        private readonly int _sampleCount;
        private readonly float _p;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="vse">Vectorization mode</param>
        /// <param name="sampleCount">Sample count per neuron per inference iteration (typically 1000 - 10000)</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public CPUDropConnectForwardPropagationFactory(
            VectorizationSizeEnum vse = VectorizationSizeEnum.VectorizationMode16,
            int sampleCount = 10000,
            float p = 0.5f)
        {
            if (sampleCount <= 0)
            {
                throw new ArgumentOutOfRangeException("sampleCount");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            _vse = vse;
            _sampleCount = sampleCount;
            _p = p;
        }

        /// <summary>
        /// Factory method
        /// </summary>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="clProvider">OpenCL provider</param>
        /// <param name="mlp">Trained MLP</param>
        /// <returns>Forward propagator</returns>
        public IForwardPropagation Create(
            IRandomizer randomizer,
            CLProvider clProvider,
            IMLP mlp)
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
