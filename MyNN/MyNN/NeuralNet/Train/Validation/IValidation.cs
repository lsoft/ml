using MyNN.NeuralNet.Structure;

namespace MyNN.NeuralNet.Train.Validation
{
    public interface IValidation
    {
        bool IsAuencoderDataSet
        {
            get;
        }

        void Validate(
            MultiLayerNeuralNetwork network,
            string epocheRoot,
            float cumulativeError,
            bool allowToSave);
    }
}