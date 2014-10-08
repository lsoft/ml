using System;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.MLP.DropConnect.ForwardPropagation.Inference;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.ForwardPropagationFactory;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.ForwardPropagation.TrainItemForward.OpenCL.CPU.Factory
{
    /// <summary>
    /// Factory for forward propagator for dropconnect backpropagation algorithm than enables CPU-OpenCL
    /// For details about dropconnect please refer http://cs.nyu.edu/~wanli/dropc/
    /// </summary>
    /// <typeparam name="T">Dropconnect layer inferencer</typeparam>
    public class CPUDropConnectForwardPropagationFactory<T> : IForwardPropagationFactory
        where T : ILayerInferencer
    {
        private readonly CLProvider _clProvider;
        private readonly VectorizationSizeEnum _vse;
        private readonly int _sampleCount;
        private readonly float _p;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="clProvider">Provider OpenCL</param>
        /// <param name="vse">Vectorization mode</param>
        /// <param name="sampleCount">Sample count per neuron per inference iteration (typically 1000 - 10000)</param>
        /// <param name="p">Probability for each weight to be ONLINE (with p = 1 it disables dropconnect and convert the model to classic backprop)</param>
        public CPUDropConnectForwardPropagationFactory(
            CLProvider clProvider,
            VectorizationSizeEnum vse = VectorizationSizeEnum.VectorizationMode16,
            int sampleCount = 10000,
            float p = 0.5f)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (sampleCount <= 0)
            {
                throw new ArgumentOutOfRangeException("sampleCount");
            }
            if (p <= 0 || p > 1)
            {
                throw new ArgumentOutOfRangeException("p");
            }

            _clProvider = clProvider;
            _vse = vse;
            _sampleCount = sampleCount;
            _p = p;
        }

        /// <summary>
        /// Factory method
        /// </summary>
        /// <param name="randomizer">Random number provider</param>
        /// <param name="mlp">Trained MLP</param>
        /// <returns>Forward propagator</returns>
        public IForwardPropagation Create(
            IRandomizer randomizer,
            IMLP mlp)
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (mlp == null)
            {
                throw new ArgumentNullException("mlp");
            }

            var forwardPropagation = new InferenceOpenCLForwardPropagation<T>(
                _vse,
                mlp,
                _clProvider,
                randomizer,
                _sampleCount,
                _p);

            return forwardPropagation;
        }
    }
}
