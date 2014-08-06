using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.Backpropagation.EpocheTrainer.DropConnect.Bit.WeightMask;
using MyNN.MLP2.Structure;
using MyNN.Randomizer;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.Tests.MLP2.MaskContainers
{
    internal class BitMaskContainerTester
    {
        public static float TestContainer(
            IDeviceChooser deviceChooser,
            IMLPConfiguration mlpConfiguration,
            float p
            )
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }
            if (mlpConfiguration == null)
            {
                throw new ArgumentNullException("mlpConfiguration");
            }

            var randomizer = new DefaultRandomizer(123);

            using (var clProvider = new CLProvider(deviceChooser, false))
            {
                var mc = new BigArrayWeightBitMaskContainer(
                    clProvider,
                    mlpConfiguration,
                    randomizer,
                    p
                    );

                var totalTotal = 0L;
                var totalOnes = 0L;

                const int IterationCount = 100000;
                for (var iter = 0; iter < IterationCount; iter++)
                {
                    mc.RegenerateMask();

                    var mask = mc.BitMask;
                    foreach (var w in mc.MaskMem)
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
