namespace MyNN.BoltzmannMachines.BinaryBinary.DBN.RBM.NegativeSampler
{
    public interface IRBMNegativeSampler
    {
        string Name
        {
            get;
        }

        void PrepareTrain(
            int batchSize);

        void GetNegativeSample(
            int batchIndex,
            int maxGibbsChainLength
            );

        void PrepareBatch();

        void BatchFinished();
    }
}
