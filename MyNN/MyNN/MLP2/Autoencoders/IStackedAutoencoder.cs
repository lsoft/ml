using MyNN.Data;
using MyNN.MLP2.ArtifactContainer;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Autoencoders
{
    public interface IStackedAutoencoder
    {
        IMLP Train(
            string sdaeName,
            IArtifactContainer rootContainer,
            IDataSet trainData,
            IDataSet validationData);
    }
}