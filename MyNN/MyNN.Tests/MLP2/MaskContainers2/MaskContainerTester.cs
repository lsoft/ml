using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.MLP.DropConnect.ForwardPropagation.WeightMaskContainer2;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.Tests.MLP2.MaskContainers2
{
    internal class MaskContainerTester
    {
        public static float TestContainer(
            Func<CLProvider, IOpenCLWeightMaskContainer2> containerProvider
            )
        {
            if (containerProvider == null)
            {
                throw new ArgumentNullException("containerProvider");
            }

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(), false))
            {
                var totalTotal = 0L;
                var totalOnes = 0L;

                var container = containerProvider(clProvider);

                const int IterationCount = 100000;

                uint previousBitMask = 0;
                for (var iter = 0; iter < IterationCount; iter++)
                {
                    container.RegenerateMask();

                    var mask = container.BitMask;

                    //проверяем, что маска изменяется
                    if (previousBitMask != 0)
                    {
                        if (previousBitMask == (uint) Math.Pow(2, 31))
                        {
                            if (mask != 1)
                            {
                                throw new InternalTestFailureException("Маска должна быть равна 1");
                            }
                        }
                        else
                        {
                            if (previousBitMask*2 != mask)
                            {
                                throw new InternalTestFailureException(
                                    "Следующая маска должна быть в два раза больше предыдущей");
                            }
                        }
                    }

                    var maskMem = container.MaskMem;

                    var masked = maskMem.Array.ConvertAll(j => (j & mask));

                    var total = masked.Length;
                    var ones = masked.Count(j => j > 0);

                    totalTotal += total;
                    totalOnes += ones;

                    previousBitMask = mask;
                }

                var result = (float) (totalOnes/(double) totalTotal);

                return result;
            }
        }
    }
}
