using System;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.Mask;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.CPU
{
    public class CPULayerPropagator : IDropoutLayerPropagator
    {
        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly IOpenCLMaskContainer _maskContainer;
        private readonly IMemLayerContainer _previousMemLayerContainer;
        private readonly IMemLayerContainer _currentMemLayerContainer;
        private readonly int _prevLayerNeuronTotalCount;
        private readonly int _currentLayerNonBiasNeuronCount;
        
        private readonly Kernel _kernel;

        public IOpenCLMaskContainer MaskContainer
        {
            get
            {
                return
                    _maskContainer;
            }
        }

        public int MaskShift
        {
            get;
            private set;
        }

        public CPULayerPropagator(
            IRandomizer randomizer,
            CLProvider clProvider,
            CPUKernelSource ks,
            IOpenCLMaskContainer maskContainer,
            IMemLayerContainer previousMemLayerContainer,
            IMemLayerContainer currentMemLayerContainer,
            IFunction activationFunction,
            VectorizationSizeEnum vse,
            int prevLayerNeuronTotalCount,
            int currentLayerNonBiasNeuronCount
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
            if (ks == null)
            {
                throw new ArgumentNullException("ks");
            }
            if (maskContainer == null)
            {
                throw new ArgumentNullException("maskContainer");
            }
            if (previousMemLayerContainer == null)
            {
                throw new ArgumentNullException("previousMemLayerContainer");
            }
            if (currentMemLayerContainer == null)
            {
                throw new ArgumentNullException("currentMemLayerContainer");
            }
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }

            _randomizer = randomizer;
            _clProvider = clProvider;
            _maskContainer = maskContainer;
            _previousMemLayerContainer = previousMemLayerContainer;
            _currentMemLayerContainer = currentMemLayerContainer;
            _prevLayerNeuronTotalCount = prevLayerNeuronTotalCount;
            _currentLayerNonBiasNeuronCount = currentLayerNonBiasNeuronCount;

            string kernelName;
            var kernelSource = ks.GetKernelSource(
                vse,
                activationFunction,
                out kernelName
                );

            _kernel = clProvider.CreateKernel(
                kernelSource,
                kernelName);
        }

        public void ComputeLayer(
            )
        {
            var maxRandomShift = this._maskContainer.MaskMem.Array.Length - _currentLayerNonBiasNeuronCount;
            this.MaskShift = _randomizer.Next(maxRandomShift);

            _maskContainer.RegenerateMask();

            _kernel
                .SetKernelArgMem(0, _previousMemLayerContainer.StateMem)
                .SetKernelArgMem(1, _currentMemLayerContainer.NetMem)
                .SetKernelArgMem(2, _currentMemLayerContainer.StateMem)
                .SetKernelArgMem(3, _currentMemLayerContainer.WeightMem)
                .SetKernelArgMem(4, this._maskContainer.MaskMem)
                .SetKernelArg(5, 4, this.MaskShift)
                .SetKernelArg(6, 4, this._maskContainer.BitMask)
                .SetKernelArg(7, 4, _prevLayerNeuronTotalCount)
                .EnqueueNDRangeKernel(
                    _currentLayerNonBiasNeuronCount
                    );
        }

        public void WaitForCalculationFinished()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }
    }
}
