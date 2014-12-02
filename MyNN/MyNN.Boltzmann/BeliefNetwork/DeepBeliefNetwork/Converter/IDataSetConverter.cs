using MyNN.Common.NewData.DataSet;

namespace MyNN.Boltzmann.BeliefNetwork.DeepBeliefNetwork.Converter
{
    public interface IDataSetConverter
    {
        IDataSet Convert(IDataSet beforeTransformation);
    }
}
