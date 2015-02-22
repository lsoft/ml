using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.MLP.Convolution.Activator
{
    public interface IFunctionActivator
    {
        void Apply(
            IFunction activationFunction,
            IReferencedSquareFloat currentNet,
            IReferencedSquareFloat currentState
            );
    }
}