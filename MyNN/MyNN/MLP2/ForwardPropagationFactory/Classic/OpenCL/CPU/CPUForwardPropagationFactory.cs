using System;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.CPU
{
    /// <summary>
    /// Factory for forward propagator for classic backpropagation algorithm than enables CPU-OpenCL
    /// </summary>
    public class CPUForwardPropagationFactory : IForwardPropagationFactory
    {
        private readonly VectorizationSizeEnum _vse;

        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="vse">Vectorization mode</param>
        public CPUForwardPropagationFactory(
            VectorizationSizeEnum vse = VectorizationSizeEnum.VectorizationMode16)
        {
            _vse = vse;
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

            return 
                new CPUForwardPropagation(
                    _vse,
                    mlp,
                    clProvider);
        }
    }
}