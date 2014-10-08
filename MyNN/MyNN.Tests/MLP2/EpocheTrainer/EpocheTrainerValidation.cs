using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.Common.ArtifactContainer;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Validation;
using MyNN.MLP.ForwardPropagation;

namespace MyNN.Tests.MLP2.EpocheTrainer
{
    internal class EpocheTrainerValidation : IValidation
    {
        public IAccuracyRecord Validate(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            IArtifactContainer epocheContainer
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
                new ClassificationAccuracyRecord(
                    epocheNumber ?? 0,
                    10,
                    5,
                    0.5f);
        }

        
    }
}
