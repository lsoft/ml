using System.Collections.Generic;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler.ParallelTempering
{
    public interface ITemperature
    {
        List<float> GetTemperatureList();
    }
}