using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.CSharp
{
    public interface ICSharpAvgPoolingLayerContainer : ICSharpLayerContainer
    {
        /// <summary>
        /// Конфигурация слоя пулинга среднего значения
        /// </summary>
        new IAvgPoolingLayerConfiguration Configuration
        {
            get;
        }
    }
}