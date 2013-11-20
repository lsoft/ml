using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.SamplerProvider
{
    public class CDBaseSamplerProvider : INegativeSamplerProvider
    {
        public string Name
        {
            get
            {
                return "Contrastive divergence provider";
            }
        }

        public virtual IRBMNegativeSampler GetNegativeSampler(IRestrictedBoltzmannMachine rbm)
        {
            return 
                new CD(rbm);
        }
    }
}
