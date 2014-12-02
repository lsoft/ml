using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.Data.Set.Item;
using MyNN.Common.OpenCLHelper;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNN.Mask.Factory;
using MyNN.MLP.Dropout.ForwardPropagation.OpenCL.CPU;
using MyNN.MLP.Dropout.ForwardPropagation.OpenCL.GPU;
using MyNN.MLP.ForwardPropagation;
using MyNN.MLP.Structure.Factory;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Layer.Factory;
using MyNN.MLP.Structure.Neuron.Factory;
using MyNN.MLP.Structure.Neuron.Function;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;

namespace MyNNConsoleApp.RefactoredForDI
{
    public class TestGPUDropoutForward
    {
        public static void DoTrain()
        {
            var randomizer = new DefaultRandomizer(123);

            var datalist = new List<IDataItem>();
            for (var cc = 0; cc < 2; cc++)
            {
                var input = new float[100];
                input.Fill(() => randomizer.Next());

                var output = new float[1];

                var di = new DataItem(
                    input,
                    output
                    );
                datalist.Add(di);
            }

            var dataset = new DataSet(datalist);

            var cpu = GetFromCPU(dataset);
            var gpu = GetFromGPU(dataset);

            if (cpu.Count != gpu.Count)
            {
                throw new Exception();
            }

            for (var cc = 0; cc < cpu.Count; cc++)
            {
                float maxDiff;
                if (!ArrayOperations.ValuesAreEqual(
                    cpu[cc].NState,
                    gpu[cc].NState,
                    1e-5f,
                    out maxDiff
                    ))
                {
                    throw new Exception();
                }
            }
        }

        private static List<ILayerState> GetFromGPU(
            DataSet dataset
            )
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }

            using (var clProvider = new CLProvider(new NvidiaOrAmdGPUDeviceChooser(true), false))
            {
                var randomizer = new DefaultRandomizer(123);

                var maskFactory = new BigArrayMaskContainerFactory(
                    randomizer,
                    clProvider
                    );

                var pcc = new GPUMaskForwardPropagatorComponentConstructor(
                    randomizer,
                    clProvider,
                    maskFactory,
                    0.5f
                    );

                var mlpFactory = new MLPFactory(
                    new LayerFactory(
                        new NeuronFactory(
                            randomizer)));

                var mlp = mlpFactory.CreateMLP(
                    "1",
                    new IFunction[]
                    {
                        null,
                        new LinearFunction(1f),
                        new LinearFunction(1f),
                    },
                    new int[]
                    {
                        100,
                        100,
                        100
                    }
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                pcc.CreateComponents(
                    mlp,
                    out containers,
                    out propagators
                    );

                var forward = new ForwardPropagation(
                    containers,
                    propagators,
                    mlp
                    );

                return
                    forward.ComputeOutput(dataset);
            }
        }

        private static List<ILayerState> GetFromCPU(
            DataSet dataset
            )
        {
            if (dataset == null)
            {
                throw new ArgumentNullException("dataset");
            }

            using (var clProvider = new CLProvider(new IntelCPUDeviceChooser(true), false))
            {
                var randomizer = new DefaultRandomizer(123);

                var maskFactory = new BigArrayMaskContainerFactory(
                    randomizer,
                    clProvider
                    );

                var pcc = new CPUMaskForwardPropagatorComponentConstructor(
                    randomizer,
                    clProvider,
                    VectorizationSizeEnum.VectorizationMode16,
                    maskFactory,
                    0.5f
                    );

                var mlpFactory = new MLPFactory(
                    new LayerFactory(
                        new NeuronFactory(
                            randomizer)));

                var mlp = mlpFactory.CreateMLP(
                    "1",
                    new IFunction[]
                    {
                        null,
                        new LinearFunction(1f),
                        new LinearFunction(1f),
                    },
                    new int[]
                    {
                        100,
                        100,
                        100
                    }
                    );

                ILayerContainer[] containers;
                ILayerPropagator[] propagators;
                pcc.CreateComponents(
                    mlp,
                    out containers,
                    out propagators
                    );

                var forward = new ForwardPropagation(
                    containers,
                    propagators,
                    mlp
                    );

                return
                    forward.ComputeOutput(dataset);
            }
        }
    }
}
