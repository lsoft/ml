﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.OutputConsole;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine
{
    public interface IAccuracyController
    {
        void AddError(
            int currentEpochNumber,
            float error
            );

        bool IsNeedToStop();
    }

    public class AccuracyController : IAccuracyController
    {
        private readonly float _errorThreshold;
        private readonly int _epochThreshold;
        private readonly List<float> _lastErrors = new List<float>();

        private int _currentEpochNumber;

        public AccuracyController(
            float errorThreshold,
            int epochThreshold
            )
        {
            _errorThreshold = errorThreshold;
            _epochThreshold = epochThreshold;
        }

        public void AddError(
            int currentEpochNumber,
            float error
            )
        {
            _lastErrors.Add(error);

            if (_lastErrors.Count > 20)
            {
                _lastErrors.RemoveRange(0, _lastErrors.Count - 20);
            }

            _currentEpochNumber = currentEpochNumber;
        }


        public bool IsNeedToStop()
        {
            var result = false;

            //определяем, что ошибка за последние 10 раундов упала меньше, чем errorThreshold за предыдущие 10
            if (_lastErrors.Count >= 20)
            {
                var avg0 = _lastErrors.Take(10).Average();
                var avg1 = _lastErrors.Skip(10).Take(10).Average();

                if (avg1 > 0)
                {
                    var d = Math.Abs((avg0 - avg1) / avg0);
                    if (d < _errorThreshold)
                    {
                        result = true;

                        ConsoleAmbientContext.Console.WriteLine(
                            "Train finished by error threshold.");
                    }
                }
            }

            if (_currentEpochNumber >= _epochThreshold)
            {
                result = true;

                ConsoleAmbientContext.Console.WriteLine(
                    "Train finished by epoch threshold.");
            }

            return result;
        }

    }
}
