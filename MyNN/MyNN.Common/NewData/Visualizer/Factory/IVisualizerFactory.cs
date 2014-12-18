namespace MyNN.Common.NewData.Visualizer.Factory
{
    public interface IVisualizerFactory
    {
        IVisualizer CreateVisualizer(
            int dataCount
            );
    }
}
