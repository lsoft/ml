using MyNN.Common.Data;
using MyNN.Common.NewData.DataSet;

namespace MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator
{
    public interface IFreeEnergyCalculator
    {
        double CalculateFreeEnergy(
            float[] weights,
            IDataSet data);
    }
}
