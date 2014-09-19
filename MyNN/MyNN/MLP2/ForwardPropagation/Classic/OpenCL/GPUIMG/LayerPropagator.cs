using System;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPUIMG.Container;
using MyNN.MLP2.Structure.Neurons.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.Classic.OpenCL.GPUIMG
{
    public class LayerPropagator : ILayerPropagator
    {
        private readonly CLProvider _clProvider;
        private readonly IImgLayerContainer _previousMemLayerContainer;
        private readonly IImgLayerContainer _currentMemLayerContainer;
        private readonly int _prevLayerNeuronTotalCount;
        private readonly int _currentLayerNonBiasNeuronCount;

        private readonly Kernel _kernel;

        public LayerPropagator(
            CLProvider clProvider,
            KernelSource ks,
            IImgLayerContainer previousMemLayerContainer,
            IImgLayerContainer currentMemLayerContainer,
            IFunction activationFunction,
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
                activationFunction,
                currentLayerNonBiasNeuronCount,
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
            const uint szLocalWorkSize = 256;
            uint szGlobalWorkSize = 64 * _clProvider.Parameters.NumComputeUnits * szLocalWorkSize;

            _kernel
                .SetKernelArgImg(0, _previousMemLayerContainer.StateMem)
                .SetKernelArgImg(1, _currentMemLayerContainer.NetMem)
                .SetKernelArgImg(2, _currentMemLayerContainer.StateMem)
                .SetKernelArgImg(3, _currentMemLayerContainer.WeightMem)
                .SetKernelArgLocalMem(4, 4 * szLocalWorkSize)
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
