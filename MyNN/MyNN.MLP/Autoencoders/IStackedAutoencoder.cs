using MyNN.Common.ArtifactContainer;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.MLP.Structure;

namespace MyNN.MLP.Autoencoders
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