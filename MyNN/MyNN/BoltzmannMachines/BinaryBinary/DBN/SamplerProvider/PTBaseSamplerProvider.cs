using System;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler.ParallelTempering;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.SamplerProvider
{
    public class PTBaseSamplerProvider : INegativeSamplerProvider
    {
        private readonly PT.TemperatureApplyRuleEnum _temperatureApplyRule;
        private readonly ITemperature _temperature;

        public PTBaseSamplerProvider(
            PT.TemperatureApplyRuleEnum temperatureApplyRule,
            ITemperature temperature
            )
        {
            if (temperature == null)
            {
                throw new ArgumentNullException("temperature");
            }

            _temperatureApplyRule = temperatureApplyRule;
            _temperature = temperature;
        }

        public string Name
        {
            get
            {
                return "Parallel tempering provider";
            }
        }

        public virtual IRBMNegativeSampler GetNegativeSampler(IRestrictedBoltzmannMachine rbm)
        {
            return
                new PT(
                    rbm,
                    _temperatureApplyRule,
                    _temperature);
        }
    }
}
