using MyNN.Data;
using MyNN.MLP2.Container;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Autoencoders
{
    public interface IStackedAutoencoder
    {
        IMLP Train(
            string sdaeName,
            IMLPContainer rootContainer,
            IDataSet trainData,
            IDataSet validationData);
    }
}