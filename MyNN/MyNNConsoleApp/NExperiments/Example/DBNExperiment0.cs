using System;
using System.Collections.Generic;
using System.Linq;
using MyNN;
using MyNN.BoltzmannMachines.BinaryBinary.DBN;
using MyNN.BoltzmannMachines.BinaryBinary.DBN.SamplerProvider;
using MyNN.Data.TrainDataProvider.Noiser;
using MyNN.Data.TrainDataProvider.Noiser.Range;
using MyNN.Data.TypicalDataProvider;
using MyNN.LearningRateController;
using MyNN.MLP2.ForwardPropagation.Classic.OpenCL.CPU;
using MyNN.MLP2.OpenCLHelper;
using MyNN.MLP2.Structure;
using MyNN.MLP2.Structure.Factory;
using MyNN.MLP2.Structure.Layer.Factory;
using MyNN.MLP2.Structure.Neurons.Factory;
using MyNN.MLP2.Structure.Neurons.Function;

using MyNN.OutputConsole;
using MyNN.Randomizer;
using Ninject;
using OpenCL.Net.Wrapper;

namespace MyNNConsoleApp.NExperiments.Example
{
    public class DBNExperiment0 : INExperiment, IDisposable
    {
        private readonly StandardKernel _kernel;
        private int _rndSeed;

        public DBNExperiment0()
        {
            _kernel = new StandardKernel(
                //new NinjectSettings
                //{
                //    UseReflectionBasedInjection = true
                //}
                );

            _rndSeed = 399317;
        }

        public void Bind()
        {
            _kernel
                .Bind<IRandomizer>()
                .To<DefaultRandomizer>()
                .InSingletonScope()
                .WithConstructorArgument(
                    "rndSeed",
                    ++_rndSeed)
                ;

            _kernel
                .Bind<ISerializationHelper>()
                .To<SerializationHelper>()
                .InSingletonScope()
                ;

            _kernel
                .Bind<IRange>()
                .To<RandomRange>()
                //not a singleton!
                ;

            #region unused INoiser bindings....

            //_kernel
            //    .Bind<INoiser>()
            //    .To<GaussNoiser>()
            //    .WhenInjectedInto<AllNoisers>()
            //    .InSingletonScope()
            //    .Named(typeof(GaussNoiser).Name)
            //    .WithConstructorArgument(
            //        "stddev",
            //        0.20f)
            //    .WithConstructorArgument(
            //        "isNeedToClamp01",
            //        false)
            //    ;

            //_kernel
            //    .Bind<INoiser>()
            //    .To<MultiplierNoiser>()
            //    .WhenInjectedInto<AllNoisers>()
            //    .InSingletonScope()
            //    .Named(typeof(MultiplierNoiser).Name)
            //    .WithConstructorArgument(
            //        "applyPercent",
            //        1f)
            //    ;

            //_kernel
            //    .Bind<INoiser>()
            //    .To<DistanceChangeNoiser>()
            //    .WhenInjectedInto<AllNoisers>()
            //    .InSingletonScope()
            //    .Named(typeof(DistanceChangeNoiser).Name)
            //    .WithConstructorArgument(
            //        "changePercent",
            //        1f)
            //    .WithConstructorArgument(
            //        "maxDistance",
            //        3)
            //    ;

            //_kernel
            //    .Bind<INoiser>()
            //    .To<SaltAndPepperNoiser>()
            //    .WhenInjectedInto<AllNoisers>()
            //    .InSingletonScope()
            //    .Named(typeof(SaltAndPepperNoiser).Name)
            //    .WithConstructorArgument(
            //        "applyPercent",
            //        0.1f)
            //    ;

            //_kernel
            //    .Bind<INoiser>()
            //    .To<ZeroMaskingNoiser>()
            //    .WhenInjectedInto<AllNoisers>()
            //    .InSingletonScope()
            //    .Named(typeof(ZeroMaskingNoiser).Name)
            //    .WithConstructorArgument(
            //        "zeroPercent",
            //        0.25f)
            //    ;

            #endregion

            _kernel
                .Bind<INoiser>()
                .To<AllNoisers>()
                .InSingletonScope()
                .WithConstructorArgument(
                    "noiserList",
                    (c) =>
                    {
                        var lambdaRandomizer = c.Kernel.Get<IRandomizer>();

                        return
                            new INoiser[]
                            {
                                new GaussNoiser(0.20f, false, new RandomRange(lambdaRandomizer)),
                                new MultiplierNoiser(lambdaRandomizer, 1f, new RandomRange(lambdaRandomizer)),
                                new DistanceChangeNoiser(lambdaRandomizer, 1f, 3, new RandomRange(lambdaRandomizer)),
                                new SaltAndPepperNoiser(lambdaRandomizer, 0.1f, new RandomRange(lambdaRandomizer)),
                                new ZeroMaskingNoiser(lambdaRandomizer, 0.25f, new RandomRange(lambdaRandomizer))
                            };
                    })
                ;

            _kernel
                .Bind<INeuronFactory>()
                .To<NeuronFactory>()
                .InSingletonScope()
                ;

            _kernel
                .Bind<ILayerFactory>()
                .To<LayerFactory>()
                .InSingletonScope()
                ;

            _kernel
                .Bind<IMLPFactory>()
                .To<MLPFactory>()
                .InSingletonScope()
                ;

        }

        public void Execute()
        {
            //var layerFactory = new LayerFactory(
            //    new NeuronFactory(
            //        _kernel.Get<IRandomizer>()));

            //

            //var mlpf = new MLPFactory(
            //    layerFactory,
            //    surgeonFactory
            //    );

            var mlpf = _kernel.Get<IMLPFactory>();

            var mlp = mlpf.CreateMLP(
                null,//"SmallDBN",
                null,
                null,
                new IFunction[]
                {
                    null,
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                });

            ConsoleAmbientContext.Console.WriteLine(mlp.GetLayerInformation());

            var autoencoder_mlp = mlpf.CreateAutoencoderMLP(
                null,//"SmallDBN",
                null,
                null,
                new IFunction[]
                {
                    null,
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                    new SigmoidFunction(1f), 
                });

            ConsoleAmbientContext.Console.WriteLine(autoencoder_mlp.GetLayerInformation());

            var trainData = MNISTDataProvider.GetDataSet(
                "_MNIST_DATABASE/mnist/trainingset/",
                1
                );
            trainData.Normalize();
            trainData.Binarize(_kernel.Get<IRandomizer>());

            using (var clProvider = new CLProvider())
            {
                var forward = new CPUForwardPropagation(
                    VectorizationSizeEnum.VectorizationMode16,
                    autoencoder_mlp,
                    clProvider);

                var result = forward.ComputeOutput(trainData);

                var vdata = new List<Pair<float[], float[]>>();
                for (var cc = 0; cc < result.Count; cc++)
                {
                    vdata.Add(
                        new Pair<float[], float[]>(
                            trainData[cc].Input,
                            result[cc].State));
                }

                var v = new MNISTVisualizer();
                v.SaveAsPairList(
                    "_.bmp",
                    vdata);

            }

            Console.ReadLine();



            //var dbn = new DeepBeliefNetwork(
            //    _kernel.Get<IRandomizer>(),
            //    28,
            //    28,
            //    300,
            //    new CDBaseSamplerProvider(),
            //    784, 200, 150, 100);

            //var trainData = MNISTDataProvider.GetDataSet(
            //    "_MNIST_DATABASE/mnist/trainingset/",
            //    2000
            //    );
            //trainData.Normalize();
            //trainData.Binarize(_kernel.Get<IRandomizer>());

            //var validationData = MNISTDataProvider.GetDataSet(
            //    "_MNIST_DATABASE/mnist/testset/",
            //    500
            //    );
            //validationData.Normalize();
            //validationData.Binarize(_kernel.Get<IRandomizer>());

            //dbn.Train(
            //    trainData,
            //    validationData,
            //    1,
            //    new LinearLearningRate(0.025f, 0.99f),
            //    -0.1f,
            //    50,
            //    "SmallDBN",
            //    1
            //    );



        }

        public void Dispose()
        {
            _kernel.Dispose();
        }
    }
}
