using System;
using MyNN.Common.Randomizer;
using MyNN.MLP.DropConnect.Inferencer.OpenCL.GPU;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Layer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.Inferencer.Factory
{
    public class GPULayerInferencerFactory : ILayerInferencerFactory
    {
        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly int _sampleCount;
        private readonly float _p;

        public GPULayerInferencerFactory(
            IRandomizer randomizer, 
            CLProvider clProvider, 
            int sampleCount, 
            float p
            )
        {
            if (randomizer == null)
            {
                throw new ArgumentNullException("randomizer");
            }
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }

            _randomizer = randomizer;
            _clProvider = clProvider;
            _sampleCount = sampleCount;
            _p = p;
        }

        public ILayerInferencer CreateLayerInferencer(
            ILayer previousLayer, 
            ILayer currentLayer, 
            IMemLayerContainer previousLayerContainer, 
            IMemLayerContainer currentLayerContainer
            )
        {
            if (previousLayer == null)
            {
                throw new ArgumentNullException("previousLayer");
            }
            if (currentLayer == null)
            {
                throw new ArgumentNullException("currentLayer");
            }
            if (previousLayerContainer == null)
            {
                throw new ArgumentNullException("previousLayerContainer");
            }
            if (currentLayerContainer == null)
            {
                throw new ArgumentNullException("currentLayerContainer");
            }

            return 
                new GPULayerInferencer(
                    _randomizer,
                    _clProvider,
                    _sampleCount,
                    previousLayer,
                    currentLayer,
                    currentLayerContainer.WeightMem,
                    previousLayerContainer.StateMem,
                    currentLayerContainer.StateMem,
                    _p
                    );

        }
    }
}