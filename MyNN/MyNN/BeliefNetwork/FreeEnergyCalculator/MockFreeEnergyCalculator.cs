using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.Data;

namespace MyNN.BeliefNetwork.FreeEnergyCalculator
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
