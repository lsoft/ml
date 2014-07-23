﻿using System;
using System.Collections.Generic;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator;
using MyNN.MLP2.Backpropagation.Validation.NLNCA.Drawer;
using MyNN.MLP2.Container;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Structure.Layer;
using MyNN.OutputConsole;

namespace MyNN.MLP2.Backpropagation.Validation
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
                "Per item error = {0}",
                accuracyRecord.PerItemError);

            return
                accuracyRecord;
        }

    }
}