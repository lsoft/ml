using MyNN.NeuralNet.Structure;
using MyNN.NeuralNet.Structure.Layers;
using MyNN.NeuralNet.Structure.Neurons;

namespace MyNN.NeuralNet.Train
{
    public delegate void MultilayerTrainProcessDelegate(
        MultiLayerNeuralNetwork network,
        string epocheRoot,
        float cumulativeError,
        bool allowToSave);
}
