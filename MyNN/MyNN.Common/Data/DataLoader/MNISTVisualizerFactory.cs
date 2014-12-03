using MyNN.Common.Data.Visualizer;
using MyNN.Common.Data.Visualizer.Factory;

namespace MyNN.Common.Data.DataLoader
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