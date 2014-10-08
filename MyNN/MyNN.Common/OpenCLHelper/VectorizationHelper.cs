using System;

namespace MyNN.Common.OpenCLHelper
{
    public class VectorizationHelper
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

        public static string GetKernelName(string kernelRoot, VectorizationSizeEnum vse)
        {
            if (kernelRoot == null)
            {
                throw new ArgumentNullException("kernelRoot");
            }

            return
                kernelRoot + GetVectorizationSize(vse).ToString();
        }
    }
}