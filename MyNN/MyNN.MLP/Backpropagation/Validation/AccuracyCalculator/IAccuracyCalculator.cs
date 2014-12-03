using System.Collections.Generic;
using MyNN.MLP.AccuracyRecord;
using MyNN.MLP.Backpropagation.Validation.Drawer;
using MyNN.MLP.Backpropagation.Validation.Drawer.Factory;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.AccuracyCalculator
{
    public interface IAccuracyCalculator
    {
        void CalculateAccuracy(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            IDrawer drawer,
            out IAccuracyRecord accuracyRecord
            );
    }
}
