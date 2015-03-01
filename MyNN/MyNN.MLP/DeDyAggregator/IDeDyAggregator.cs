namespace MyNN.MLP.DeDyAggregator
{
    public interface IDeDyAggregator
    {
        int TotalNeuronCount
        {
            get;
        }

        void Aggregate(
            );

        void ClearAndWrite();
    }
}