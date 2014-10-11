using MyNN.Common.Data;
using MyNN.Common.Data.Set;

namespace MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator
{
    public interface IFreeEnergyCalculator
    {
        double CalculateFreeEnergy(
            float[] weights,
            IDataSet data);
    }
}
