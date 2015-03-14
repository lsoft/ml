using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.CSharp
{
    public class CSharpAvgPoolingLayerContainer : CSharpLayerContainer, ICSharpAvgPoolingLayerContainer
    {
        public new IAvgPoolingLayerConfiguration Configuration
        {
            get;
            private set;
        }


        public CSharpAvgPoolingLayerContainer(IAvgPoolingLayerConfiguration layerConfiguration)
            : base(layerConfiguration)
        {
            Configuration = layerConfiguration;
        }
    }
}