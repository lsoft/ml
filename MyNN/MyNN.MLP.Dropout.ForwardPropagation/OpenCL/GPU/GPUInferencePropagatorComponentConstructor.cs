using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.Mask.Factory;
using MyNN.MLP.Structure;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.GPU
{
    public class GPUInferencePropagatorComponentConstructor : GPUPropagatorComponentConstructor
    {
        public GPUInferencePropagatorComponentConstructor(
            IRandomizer randomizer,
            CLProvider clProvider,
            IOpenCLMaskContainerFactory maskContainerFactory,
            float p
            )
            : base(randomizer, clProvider, maskContainerFactory, p)
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
            //это первый скрытый слой?
            var isFirstHiddenLayer = layerIndex == 1;

            if (isFirstHiddenLayer)
            {
                //на первом скрытом слое веса не уменьшаются, так как на входном слое нейроны не выпадают
                //естесственно, независимо от маски, так как мы в процессе выведения, а не обучения
                zeroValue0 = 1f;
                oneValue0 = 1f;
                zeroValue1 = 1f;
                oneValue1 = 1f;
            }
            else
            {
                //на не первом скрытом слое и на выходном слое надо уменьшить в два раза net  и не влиять на state
                //естесственно, независимо от маски, так как мы в процессе выведения, а не обучения
                zeroValue0 = 0.5f;
                oneValue0 = 0.5f;
                zeroValue1 = 1f;
                oneValue1 = 1f;
            }
        }
    }
}