using MyNN.Common.ArtifactContainer;
using MyNN.Common.NewData.DataSet;

namespace MyNN.Boltzmann.BeliefNetwork.RestrictedBoltzmannMachine.Container
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

        void SetInput(float[] input);

        void SetHidden(float[] hidden);

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