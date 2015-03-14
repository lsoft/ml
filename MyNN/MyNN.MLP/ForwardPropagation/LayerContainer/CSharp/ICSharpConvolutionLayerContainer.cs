using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.CSharp
{
    public interface ICSharpConvolutionLayerContainer : ICSharpLayerContainer
    {
        /// <summary>
        /// ������������ ���� �������
        /// </summary>
        new IConvolutionLayerConfiguration Configuration
        {
            get;
        }
    }
}