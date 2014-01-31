using System;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPU;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagationFactory.Classic.OpenCL.GPU
{
    /// <summary>
    /// Factory for forward propagator for classic backpropagation algorithm than enables GPU-OpenCL
    /// </summary>
    public class GPUForwardPropagationFactory : IForwardPropagationFactory
    {

        public GPUForwardPropagationFactory()
        {
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
                new GPUForwardPropagation(
                    mlp,
                    clProvider);
        }
    }
}