using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.DBNInfo.WeightLoader
{
    public interface IWeightLoader
    {
        void LoadWeights(ILayer layer);
    }
}
