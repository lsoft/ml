﻿using System;

namespace MyNN.Common.LearningRateController
{
    public class ConstLearningRate : ILearningRate
    {
        private readonly float _startLearningRate;

        public ConstLearningRate(float learningRate)
        {
            if (learningRate <= 0)
            {
                throw new ArgumentException("learningRate");
            }

            _startLearningRate = learningRate;
        }

        public float GetLearningRate(int epocheNumber)
        {
            return
                _startLearningRate;
        }

    }
}
