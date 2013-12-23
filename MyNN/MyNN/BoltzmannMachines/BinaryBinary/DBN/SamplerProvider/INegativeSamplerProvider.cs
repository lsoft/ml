using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.SamplerProvider
{
    public interface INegativeSamplerProvider
    {
        string Name
        {
            get;
        }

        IRBMNegativeSampler GetNegativeSampler(IRestrictedBoltzmannMachine rbm);
    }
}
