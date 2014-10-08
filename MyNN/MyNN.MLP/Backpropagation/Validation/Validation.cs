using System;
using System.Collections.Generic;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.OutputConsole;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation
{
    public class Validation : IValidation
    {
        private readonly IAccuracyCalculator _accuracyCalculator;
        private readonly IDrawer _drawer;

        public Validation(
            IAccuracyCalculator accuracyCalculator,
            IDrawer drawer
            )
        {
            if (accuracyCalculator == null)
            {
                throw new ArgumentNullException("accuracyCalculator");
            }
            //drawer allowed to be null

            _accuracyCalculator = accuracyCalculator;
            _drawer = drawer;
        }

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

            List<ILayerState> netResults;
            IAccuracyRecord accuracyRecord;
            _accuracyCalculator.CalculateAccuracy(
                forwardPropagation,
                epocheNumber,
                out netResults,
                out accuracyRecord
                );

            if (_drawer != null)
            {
                _drawer.Draw(
                    epocheContainer,
                    epocheNumber,
                    netResults
                    );
            }

            ConsoleAmbientContext.Console.WriteLine(
                "Per item error (before regularization) = {0}",
                accuracyRecord.PerItemError);

            return
                accuracyRecord;
        }

    }
}
