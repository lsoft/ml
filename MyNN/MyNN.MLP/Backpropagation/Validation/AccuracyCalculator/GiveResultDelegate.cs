using MyNN.Common.NewData.Item;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Backpropagation.Validation.AccuracyCalculator
{
    public delegate void GiveResultDelegate(
        ILayerState modelResult,
        IDataItem origData
        );
}