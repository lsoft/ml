using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.LearningRateController
{
    public class LinearLearningRate : ILearningRate
    {
        private readonly float _startLearningRate;
        private readonly float _coef;
        
        public LinearLearningRate(
            float startLearningRate,
            float coef)
        {
            if (startLearningRate <= 0)
            {
                throw new ArgumentException("startLearningRate");
            }

            if (coef <= 0)
            {
                throw new ArgumentException("coef");
            }

            if (coef > 1)
            {
                throw new ArgumentException("coef");
            }

            _startLearningRate = startLearningRate;
            _coef = coef;
        }


        public float GetLearningRate(int epocheNumber)
        {
            return
                _startLearningRate * (float) Math.Pow(_coef, epocheNumber);
        }

    }
}
