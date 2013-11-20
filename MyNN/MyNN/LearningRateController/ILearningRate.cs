using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.LearningRateController
{
    public interface ILearningRate
    {
        float GetLearningRate(int epocheNumber);
    }
}
