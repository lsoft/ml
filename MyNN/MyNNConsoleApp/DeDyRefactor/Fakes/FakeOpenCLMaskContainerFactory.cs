using System;
using MyNN.Mask;
using MyNN.Mask.Factory;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.DeDyRefactor.Fakes
{
    internal class FakeOpenCLMaskContainerFactory : IOpenCLMaskContainerFactory
    {
        private readonly CLProvider _clProvider;

        public FakeOpenCLMaskContainerFactory(
            CLProvider clProvider
            )
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            _clProvider = clProvider;
        }

        public IOpenCLMaskContainer CreateContainer(long arraySize, float p)
        {
            
            return 
                new FakeOpenCLMaskContainer(
                    _clProvider,
                    arraySize
                    );
        }
    }
}