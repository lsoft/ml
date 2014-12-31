using MyNN.Common.NewData.DataSet;

namespace MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator
{
    public interface IFreeEnergyCalculator
    {
        double CalculateFreeEnergy(
            float[] weights,
            float[] visibleBiases,
            float[] hiddenBiases,
            IDataSet data
            );
    }
}
