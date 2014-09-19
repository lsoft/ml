using System;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.Container;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU
{
    public class CPULayerPropagator : ILayerPropagator
    {
        private readonly CLProvider _clProvider;
        private readonly IMemLayerContainer _previousMemLayerContainer;
        private readonly IMemLayerContainer _currentMemLayerContainer;
        private readonly int _prevLayerNeuronTotalCount;
        private readonly int _currentLayerNonBiasNeuronCount;
        
        private readonly Kernel _kernel;

        public CPULayerPropagator(
            CLProvider clProvider,
            CPUKernelSource ks,
            IMemLayerContainer previousMemLayerContainer,
            IMemLayerContainer currentMemLayerContainer,
            IFunction activationFunction,
            VectorizationSizeEnum vse,
            int prevLayerNeuronTotalCount,
            int currentLayerNonBiasNeuronCount
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
            _kernel
                .SetKernelArgMem(0, _previousMemLayerContainer.StateMem)
                .SetKernelArgMem(1, _currentMemLayerContainer.NetMem)
                .SetKernelArgMem(2, _currentMemLayerContainer.StateMem)
                .SetKernelArgMem(3, _currentMemLayerContainer.WeightMem)
                .SetKernelArg(4, 4, _prevLayerNeuronTotalCount)
                .EnqueueNDRangeKernel(_currentLayerNonBiasNeuronCount);
        }

        public void WaitForCalculationFinished()
        {
            // Make sure we're done with everything that's been requested before
            _clProvider.QueueFinish();
        }
    }
}
