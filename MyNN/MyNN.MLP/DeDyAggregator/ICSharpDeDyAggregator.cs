namespace MyNN.MLP.DeDyAggregator
{
    public interface ICSharpDeDyAggregator : IDeDyAggregator
    {
        float[] DeDz
        {
            get;
        }

        float[] DeDy
        {
            get;
        }
    }
}