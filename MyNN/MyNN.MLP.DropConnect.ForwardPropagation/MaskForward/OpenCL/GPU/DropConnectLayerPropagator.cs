using System;
using MyNN.Mask;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.DropConnect.ForwardPropagation.MaskForward.OpenCL.GPU
{
    public class DropConnectLayerPropagator : IDropConnectLayerPropagator
    {
        private readonly CLProvider _clProvider;
        private readonly IOpenCLMaskContainer _maskContainer;
        private readonly IMemLayerContainer _previousMemLayerContainer;
        private readonly IMemLayerContainer _currentMemLayerContainer;
        private readonly int _prevLayerNeuronTotalCount;
        private readonly int _currentLayerTotalNeuronCount;

        private readonly Kernel _kernel;

        public IOpenCLMaskContainer MaskContainer
        {
            get
            {
                return
                    _maskContainer;
            }
        }

        public DropConnectLayerPropagator(
            CLProvider clProvider,
            KernelSource ks,
            IOpenCLMaskContainer maskContainer,
            IMemLayerContainer previousMemLayerContainer,
            IMemLayerContainer currentMemLayerContainer,
            IFunction activationFunction,
            int prevLayerNeuronTotalCount,
            int currentLayerTotalNeuronCount
            )
        {
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

            _clProvider = clProvider;
            _maskContainer = maskContainer;
            _previousMemLayerContainer = previousMemLayerContainer;
            _currentMemLayerContainer = currentMemLayerContainer;
            _prevLayerNeuronTotalCount = prevLayerNeuronTotalCount;
            _currentLayerTotalNeuronCount = currentLayerTotalNeuronCount;

            string kernelName;
            var kernelSource = ks.GetKernelSource(
                activationFunction,
                currentLayerTotalNeuronCount,
                prevLayerNeuronTotalCount,
                out kernelName
                );

            _kernel = clProvider.CreateKernel(
                kernelSource,
                kernelName);
        }

        public void ComputeLayer(
            )
        {
            _maskContainer.RegenerateMask();

            const uint szLocalWorkSize = 256;
            uint szGlobalWorkSize = 64 * _clProvider.Parameters.NumComputeUnits * szLocalWorkSize;

            _kernel
                .SetKernelArgMem(0, _previousMemLayerContainer.StateMem)
                .SetKernelArgMem(1, _currentMemLayerContainer.NetMem)
                .SetKernelArgMem(2, _currentMemLayerContainer.StateMem)
                .SetKernelArgMem(3, _currentMemLayerContainer.WeightMem)
                .SetKernelArgMem(4, _maskContainer.MaskMem)
                .SetKernelArg(5, 4, _maskContainer.BitMask)
                .SetKernelArgLocalMem(6, 4 * szLocalWorkSize)
                .SetKernelArgMem(7, _currentMemLayerContainer.BiasMem)
                .EnqueueNDRangeKernel(
                    new[]
                        {
                            szGlobalWorkSize
                        }
                    , new[]
                        {
                            szLocalWorkSize
                        }
                    );
        }

        public void WaitForCalculationFinished()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }

    }
}
