using MyNN.Data;
using MyNN.MLP2.ArtifactContainer;

namespace MyNN.BeliefNetwork.RestrictedBoltzmannMachine.Container
{
    public interface IContainer
    {
        int VisibleNeuronCount
        {
            get;
        }

        int HiddenNeuronCount
        {
            get;
        }

        void SetTrainItem(float[] input);

        void ClearNabla();

        void CalculateNabla();

        void UpdateWeights(
            int batchSize,
            float learningRate);

        float GetError();

        void Save(IArtifactContainer container);

        double CalculateFreeEnergy(
            IDataSet data);
    }
}