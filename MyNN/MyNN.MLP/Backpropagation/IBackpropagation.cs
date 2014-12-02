using MyNN.Common.Data.TrainDataProvider;
using MyNN.Common.NewData.DataSetProvider;
using MyNN.MLP.AccuracyRecord;

namespace MyNN.MLP.Backpropagation
{
    public interface IBackpropagation
    {
        IAccuracyRecord Train(IDataSetProvider dataSetProvider);
    }
}