using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.ForwardPropagation.LayerContainer.CSharp
{
    public interface ICSharpAvgPoolingLayerContainer : ICSharpLayerContainer
    {
        /// <summary>
        /// ������������ ���� ������� �������� ��������
        /// </summary>
        new IAvgPoolingLayerConfiguration Configuration
        {
            get;
        }
    }
}