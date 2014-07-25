using System.IO;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.ArtifactContainer
{
    public interface IArtifactContainer : IArtifactReadContainer
    {
        void SaveMLP(
            IMLP mlp,
            IAccuracyRecord accuracyRecord
            );

        Stream GetWriteStreamForResource(string resourceName);

        IArtifactContainer GetChildContainer(string containerName);
        
        void Clear();

        void DeleteResource(string resourceName);
    }
}
