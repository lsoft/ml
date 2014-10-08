using System;
using MyNN.Common.Randomizer;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure;

namespace MyNN.MLP.ForwardPropagationFactory
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
                new ForwardPropagation.ForwardPropagation(
                    containers,
                    propagators,
                    mlp
                    );
        }
    }
}