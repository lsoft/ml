using MyNN.Data;
using MyNN.MLP2.Structure;

namespace MyNN.MLP2.Autoencoders
{
    public interface IStackedAutoencoder
    {
        IMLP CombinedNet
        {
            get;
        }

        IMLP Train(
            string root,
            DataSet trainData,
            DataSet validationData);
    }
}