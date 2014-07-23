﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.Container;
using MyNN.MLP2.ForwardPropagation;

namespace MyNN.Tests.MLP2.EpocheTrainer
{
    internal class EpocheTrainerValidation : IValidation
    {
        public IAccuracyRecord Validate(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            IMLPContainer epocheContainer
            )
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }
            if (epocheContainer == null)
            {
                throw new ArgumentNullException("epocheContainer");
            }

            return
                null;
        }
    }
}
