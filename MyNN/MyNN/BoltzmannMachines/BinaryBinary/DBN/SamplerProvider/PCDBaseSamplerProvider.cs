using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.SamplerProvider
{
    public class PCDBaseSamplerProvider : INegativeSamplerProvider
    {
        public string Name
        {
            get
            {
                return "Persistent contrastive divergence provider";
            }
        }

        public virtual IRBMNegativeSampler GetNegativeSampler(IRestrictedBoltzmannMachine rbm)
        {
            return 
                new PCD(rbm);
        }
    }
}
