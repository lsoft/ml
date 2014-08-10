using System;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.ForwardPropagation.Classic;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagationFactory.Classic
{
    /// <summary>
    /// Factory for forward propagator for classic backpropagation algorithm
    /// </summary>
    public class ForwardPropagationFactory : IForwardPropagationFactory
    {
        private readonly IPropagatorComponentConstructor _componentConstructor;

        /// <summary>
        /// Constructor
        /// </summary>
        public ForwardPropagationFactory(
            IPropagatorComponentConstructor componentConstructor
            )
        {
            if (componentConstructor == null)
            {
                throw new ArgumentNullException("componentConstructor");
            }

            _componentConstructor = componentConstructor;
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

            ILayerContainer[] containers;
            ILayerPropagator[] propagators;
            _componentConstructor.CreateComponents(
                mlp,
                out containers,
                out propagators);

            return
                new ForwardPropagation2(
                    containers,
                    propagators,
                    mlp
                    );
        }
    }
}