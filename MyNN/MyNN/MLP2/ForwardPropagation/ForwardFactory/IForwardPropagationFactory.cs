using MyNN.MLP2.Randomizer;
using MyNN.MLP2.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.ForwardFactory
{
    public interface IForwardPropagationFactory
    {
        IForwardPropagation Create(
            IRandomizer randomizer,
            CLProvider clProvider,
            MLP mlp);
    }
}