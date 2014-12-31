using System;
using MyNN.Common.NewData.DataSet;

namespace MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator
{
    public class MockFreeEnergyCalculator : IFreeEnergyCalculator
    {
        public double CalculateFreeEnergy(
            float[] weights,
            float[] visibleBiases,
            float[] hiddenBiases,
            IDataSet data
            )
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
            }
            if (visibleBiases == null)
            {
                throw new ArgumentNullException("visibleBiases");
            }
            if (hiddenBiases == null)
            {
                throw new ArgumentNullException("hiddenBiases");
            }
            if (data == null)
            {
                throw new ArgumentNullException("data");
            }

            return
                double.NaN;
        }
    }
}
