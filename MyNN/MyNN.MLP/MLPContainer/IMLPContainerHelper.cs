using MyNN.Common.ArtifactContainer;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Structure;

namespace MyNN.MLP.MLPContainer
{
    public interface IMLPContainerHelper
    {
        void SaveMLP(
            IArtifactContainer artifactContainer,
            IMLP mlp,
            IAccuracyRecord accuracyRecord
            );

        IMLP Load(
            IArtifactContainer artifactContainer, 
            string mlpName
            );
    }
}
