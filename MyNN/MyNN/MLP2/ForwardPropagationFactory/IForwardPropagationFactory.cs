using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagationFactory
{
    public interface IForwardPropagationFactory
    {
        IForwardPropagation Create(
            IRandomizer randomizer,
            CLProvider clProvider,
            MLP mlp);
    }
}