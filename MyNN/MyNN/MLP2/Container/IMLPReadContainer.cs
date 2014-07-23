using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Container
{
    public interface IMLPReadContainer
    {
        IMLP Load(string mlpName);

        IMLP DeepClone(IMLP mlp);
    }
}