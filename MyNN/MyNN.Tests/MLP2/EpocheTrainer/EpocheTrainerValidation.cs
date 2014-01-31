using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.MLP2.Backpropagation.Validation;
using MyNN.MLP2.ForwardPropagation;

namespace MyNN.Tests.MLP2.EpocheTrainer
{
    internal class EpocheTrainerValidation : IValidation
    {
        public float Validate(
            IForwardPropagation forwardPropagation,
            string epocheRoot,
            bool allowToSave)
        {
            if (forwardPropagation == null)
            {
                throw new ArgumentNullException("forwardPropagation");
            }
            if (epocheRoot == null)
            {
                throw new ArgumentNullException("epocheRoot");
            }

            return 1f;
        }
    }
}
