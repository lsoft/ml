using MyNN.MLP2.Structure;

namespace MyNN.MLP2.ArtifactContainer
{
    public interface IArtifactReadContainer
    {
        IMLP Load(string mlpName);

        IMLP DeepClone(IMLP mlp);
    }
}