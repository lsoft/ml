using System;
using System.Collections.Generic;
using MyNN.Common.ArtifactContainer;
using MyNN.Common.OutputConsole;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Backpropagation.Validation.Drawer.Factory;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation
{
    public class Validation : IValidation
    {
        private readonly IAccuracyCalculator _accuracyCalculator;
        private readonly IDrawerFactory _drawerFactory;

        public Validation(
            IAccuracyCalculator accuracyCalculator,
            IDrawerFactory drawerFactory
            )
        {
            if (accuracyCalculator == null)
            {
                throw new ArgumentNullException("accuracyCalculator");
            }
            //drawerFactory allowed to be null

            _accuracyCalculator = accuracyCalculator;
            _drawerFactory = drawerFactory;
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

            var drawer = _drawerFactory != null
                ? _drawerFactory.CreateDrawer(
                    epocheContainer,
                    epocheNumber
                    )
                : null;

            IAccuracyRecord accuracyRecord;
            _accuracyCalculator.CalculateAccuracy(
                forwardPropagation,
                epocheNumber,
                drawer,
                out accuracyRecord
                );

            ConsoleAmbientContext.Console.WriteLine(
                "Per item error (before regularization) = {0}",
                accuracyRecord.PerItemError);

            return
                accuracyRecord;
        }

    }
}
