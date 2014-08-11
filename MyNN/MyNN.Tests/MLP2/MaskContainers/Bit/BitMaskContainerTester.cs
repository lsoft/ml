﻿using System;
using System.Linq;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.WeightMask;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.Tests.MLP2.MaskContainers.Bit
{
    internal class BitMaskContainerTester
    {
        public static float TestContainer(
            Func<CLProvider, IOpenCLWeightBitMaskContainer> containerProvider
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
                for (var iter = 0; iter < IterationCount; iter++)
                {
                    container.RegenerateMask();

                    var mask = container.BitMask;
                    foreach (var w in container.MaskMem)
                    {
                        if (w != null)
                        {
                            var total = w.Array.Length;
                            var ones = w.Array.Count(j => (j & mask) > 0);

                            totalTotal += total;
                            totalOnes += ones;
                        }
                    }
                }

                var result = (float) (totalOnes/(double) totalTotal);

                return result;
            }
        }
    }
}