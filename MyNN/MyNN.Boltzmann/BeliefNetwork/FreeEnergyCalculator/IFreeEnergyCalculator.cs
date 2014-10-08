using MyNN.Common.Data;

namespace MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator
{
    public interface IFreeEnergyCalculator
    {
        double CalculateFreeEnergy(
            float[] weights,
            IDataSet data);
    }
}
