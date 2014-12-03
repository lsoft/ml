using MyNN.Common.Data.Set.Item;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.AccuracyCalculator
{
    public delegate void GiveResultDelegate(
        ILayerState modelResult,
        IDataItem origData
        );
}