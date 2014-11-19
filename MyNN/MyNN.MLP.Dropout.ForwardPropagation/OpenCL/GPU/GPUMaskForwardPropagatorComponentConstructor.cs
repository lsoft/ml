using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.Mask.Factory;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.GPU
{
    public class GPUMaskForwardPropagatorComponentConstructor : GPUPropagatorComponentConstructor
    {
        public GPUMaskForwardPropagatorComponentConstructor(
            IRandomizer randomizer,
            CLProvider clProvider,
            IOpenCLMaskContainerFactory maskContainerFactory,
            float p
            ) : base(randomizer, clProvider, maskContainerFactory, p)
        {
        }

        protected override void GetMaskCoefficient(
            IMLP mlp, 
            int layerIndex, 
            out float zeroValue0,
            out float oneValue0,
            out float zeroValue1,
            out float oneValue1
            )
        {
            var layerCount = mlp.Layers.Length;

            //это выходной (последний) слой?
            var isOutputLayer = layerIndex == (layerCount - 1);

            if (isOutputLayer)
            {
                //на выходном слое в процессе обучения дропаут не работает
                zeroValue0 = 1f;
                oneValue0 = 1f;
                zeroValue1 = 1f;
                oneValue1 = 1f;
            }
            else
            {
                //на скрытом слое дропаут работает как положено
                zeroValue0 = 0f;
                oneValue0 = 1f;
                zeroValue1 = 0f;
                oneValue1 = 1f;
            }
        }
    }
}