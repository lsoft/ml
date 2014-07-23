using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.AccuracyRecord;
using MyNN.MLP2.ForwardPropagation;
using MyNN.MLP2.Structure.Layer;

namespace MyNN.MLP2.Backpropagation.Validation.AccuracyCalculator
{
    public interface IAccuracyCalculator
    {
        void CalculateAccuracy(
            IForwardPropagation forwardPropagation,
            int? epocheNumber,
            out List<ILayerState> netResults,
            out IAccuracyRecord accuracyRecord
            );
    }
}
