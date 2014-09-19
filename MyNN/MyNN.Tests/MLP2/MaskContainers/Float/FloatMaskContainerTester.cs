using System;
using System.Linq;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.WeightMask;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Float.WeightMask;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNN.Tests.MLP2.MaskContainers.Float
{
    internal class FloatMaskContainerTester
    {
        public static float TestContainer(
            IDeviceChooser deviceChooser,
            Func<CLProvider, IOpenCLWeightMaskContainer> containerProvider
            )
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }
            if (containerProvider == null)
            {
                throw new ArgumentNullException("containerProvider");
            }

            using (var clProvider = new CLProvider(deviceChooser, false))
            {
                var totalTotal = 0L;
                var totalOnes = 0L;

                var container = containerProvider(clProvider);

                const int IterationCount = 50000;
                for (var iter = 0; iter < IterationCount; iter++)
                {
                    container.RegenerateMask();

                    foreach (var w in container.MaskMem)
                    {
                        if (w != null)
                        {
                            var total = w.Array.Length;
                            var ones = w.Array.Count(j => Math.Abs(j - 1f) < float.Epsilon );

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
