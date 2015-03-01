using System;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Randomizer;
using MyNN.Mask;
using MyNN.MLP.ForwardPropagation.LayerContainer.OpenCL.Mem;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP.Dropout.ForwardPropagation.OpenCL.GPU
{
    public class GPULayerPropagator : IDropoutLayerPropagator
    {
        private readonly IRandomizer _randomizer;
        private readonly CLProvider _clProvider;
        private readonly IOpenCLMaskContainer _maskContainer;
        private readonly IMemLayerContainer _previousMemLayerContainer;
        private readonly IMemLayerContainer _currentMemLayerContainer;
        private readonly float _zeroValue0;
        private readonly float _oneValue0;
        private readonly float _zeroValue1;
        private readonly float _oneValue1;

        private readonly Kernel _kernel;

        private int _maskChanged = 0;

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

        public GPULayerPropagator(
            IRandomizer randomizer,
            CLProvider clProvider,
            GPUKernelSource ks,
            IOpenCLMaskContainer maskContainer,
            IMemLayerContainer previousMemLayerContainer,
            IMemLayerContainer currentMemLayerContainer,
            IFunction activationFunction,
            float zeroValue0,
            float oneValue0,
            float zeroValue1,
            float oneValue1
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
            _zeroValue0 = zeroValue0;
            _oneValue0 = oneValue0;
            _zeroValue1 = zeroValue1;
            _oneValue1 = oneValue1;

            string kernelName;
            var kernelSource = ks.GetKernelSource(
                activationFunction,
                currentMemLayerContainer.Configuration.TotalNeuronCount,
                previousMemLayerContainer.Configuration.TotalNeuronCount,
                out kernelName
                );

            _kernel = clProvider.CreateKernel(
                kernelSource,
                kernelName);
        }

        public void ComputeLayer(
            )
        {
            var maxRandomShift = 
                this._maskContainer.MaskMem.Array.Length - _currentMemLayerContainer.Configuration.TotalNeuronCount;
            
            this.MaskShift = _randomizer.Next(maxRandomShift);

            if (++_maskChanged > 100)
            {
                _maskContainer.RegenerateMask();

                _maskChanged = 0;
            }

            const uint szLocalWorkSize = 256;
            uint szGlobalWorkSize = 64 * _clProvider.Parameters.NumComputeUnits * szLocalWorkSize;

            _kernel
                .SetKernelArgMem(0, _previousMemLayerContainer.StateMem)
                .SetKernelArgMem(1, _currentMemLayerContainer.NetMem)
                .SetKernelArgMem(2, _currentMemLayerContainer.StateMem)
                .SetKernelArgMem(3, _currentMemLayerContainer.WeightMem)
                .SetKernelArgMem(4, this._maskContainer.MaskMem)
                .SetKernelArg(5, 4, this.MaskShift)
                .SetKernelArg(6, 4, this._maskContainer.BitMask)
                .SetKernelArg(7, 4, _zeroValue0)
                .SetKernelArg(8, 4, _oneValue0)
                .SetKernelArg(9, 4, _zeroValue1)
                .SetKernelArg(10, 4, _oneValue1)
                .SetKernelArgLocalMem(11, 4 * szLocalWorkSize)
                .SetKernelArgMem(12, _currentMemLayerContainer.BiasMem)
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
