using System;
using MyNN.Common.Data;
using MyNN.Common.NewData.DataSet;

namespace MyNN.Boltzmann.BeliefNetwork.FreeEnergyCalculator
{
    public class MockFreeEnergyCalculator : IFreeEnergyCalculator
    {
        public double CalculateFreeEnergy(
            float[] weights, 
            IDataSet data)
        {
            if (weights == null)
            {
                throw new ArgumentNullException("weights");
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
