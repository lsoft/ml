using System;

namespace MyNN.MLP2.OpenCL
{
    internal class VectorizationHelper
    {
        public static int GetVectorizationSize(VectorizationSizeEnum vse)
        {
            switch (vse)
            {
                case VectorizationSizeEnum.NoVectorization:
                    return 1;
                case VectorizationSizeEnum.VectorizationMode4:
                    return 4;
                case VectorizationSizeEnum.VectorizationMode16:
                    return 16;
                default:
                    throw new ArgumentOutOfRangeException("vse");
            }
        }

        public static string GetKernelName(VectorizationSizeEnum vse)
        {
            switch (vse)
            {
                case VectorizationSizeEnum.NoVectorization:
                    return "ComputeLayerKernel1";
                case VectorizationSizeEnum.VectorizationMode4:
                    return "ComputeLayerKernel4";
                case VectorizationSizeEnum.VectorizationMode16:
                    return "ComputeLayerKernel16";
                default:
                    throw new ArgumentOutOfRangeException("vse");
            }
        }
    }
}