using MyNN.Common.Randomizer;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure;

namespace MyNN.MLP.ForwardPropagationFactory
{
    public interface IForwardPropagationFactory
    {
        IForwardPropagation Create(
            IRandomizer randomizer,
            IMLP mlp);
    }
}