using MyNN.Common.NewData.Visualizer;
using MyNN.Common.NewData.Visualizer.Factory;

namespace MyNN.Common.NewData.MNIST
{
    public class MNISTVisualizerFactory : IVisualizerFactory
    {
        public IVisualizer CreateVisualizer(
            int dataCount
            )
        {
            return
                new MNISTVisualizer(dataCount);

        }
    }
}