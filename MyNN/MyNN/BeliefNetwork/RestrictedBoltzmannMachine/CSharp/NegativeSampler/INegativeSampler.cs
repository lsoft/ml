namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.CSharp.NegativeSampler
{
    public interface INegativeSampler
    {
        string Name
        {
            get;
        }

        void PrepareTrain(
            int batchSize);

        void PrepareBatch();

        void CalculateNegativeSample(
            int indexIntoBatch,
            int maxGibbsChainLength
            );

        void BatchFinished();
    }
}
