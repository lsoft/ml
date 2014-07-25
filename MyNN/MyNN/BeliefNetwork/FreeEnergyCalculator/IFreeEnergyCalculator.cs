using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.Container;
using MyNN.Data;

namespace MyNN.BeliefNetwork.FreeEnergyCalculator
{
    public interface IFreeEnergyCalculator
    {
        double CalculateFreeEnergy(
            float[] weights,
            IDataSet data);
    }
}
