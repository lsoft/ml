using System;
using System.Collections.Generic;

namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler.ParallelTempering
{
    public class LinearTemperature : ITemperature
    {
        private readonly int _temperatureCount;
        private readonly float _highTemperature;

        public LinearTemperature(
            int temperatureCount,
            float highTemperature = 0f)
        {
            #region validate

            if (temperatureCount < 1)
            {
                throw new ArgumentException("temperatureCount");
            }

            if (highTemperature < 0f || highTemperature >= 1f)
            {
                throw new ArgumentException("highTemperature");
            }

            #endregion

            _temperatureCount = temperatureCount;
            _highTemperature = highTemperature;
        }

        public List<float> GetTemperatureList()
        {
            var result = new List<float>();

            for (var ti = 0; ti <= _temperatureCount; ti++)
            {
                float temperature = ti * ((1f - _highTemperature) / this._temperatureCount) + _highTemperature;

                result.Add(temperature);
            }

            return result;
        }
    }
}
