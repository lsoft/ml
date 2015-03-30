using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MyNN.Common.Other;
using MyNN.MLP.Classic.ForwardPropagation.CSharp;
using MyNN.MLP.Convolution.Activator;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.KernelBiasContainer;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.ForwardPropagation.LayerContainer.CSharp;
using MyNN.MLP.Structure.Layer;
using MyNN.MLP.Structure.Neuron;
using MyNN.MLP.Structure.Neuron.Function;

namespace MyNN.Tests.MLP2.Forward.Convolution.CSharp
{
    [TestClass]
    public class FullConnectedConvolutionFixture
    {
        [TestMethod]
        public void WithSingleFeatureMapTest()
        {
            //configuration

            const int previousLayerWidth = 5;
            const int previousLayerHeight= 5;
            const int previuosLayerNeuronCount = previousLayerWidth*previousLayerHeight;

            const int kernelSize = 2;
            const int weightCount = kernelSize*kernelSize;

            const int featureMapCount = 1;
            const int biasCount = featureMapCount;

            const int currentLayerWidth = previousLayerWidth - kernelSize + 1;
            const int currentLayerHeight= previousLayerHeight - kernelSize + 1;
            const int currentLayerNeuronCount = currentLayerWidth * currentLayerHeight;

            const float neuronValue = 2f;
            const float biasValue = 1f;
            const float weightValue = 3f;

            const float resultValue = kernelSize * kernelSize * weightValue * neuronValue + biasValue;

            //dependenced objects

            ICSharpLayerContainer previousLayerContainer = new CSharpLayerContainer(
                new LayerConfiguration(
                    null,
                    new Dimension(2, previousLayerWidth, previousLayerHeight),
                    previuosLayerNeuronCount,
                    previuosLayerNeuronCount,
                    new INeuronConfiguration[]
                    {
                        
                    }
                    )
                );

            ICSharpConvolutionLayerContainer currentLayerContainer = new CSharpConvolutionLayerContainer(
                new ConvolutionLayerConfiguration(
                    new LinearFunction(1f),
                    new Dimension(2, currentLayerWidth, currentLayerHeight),
                    featureMapCount,
                    new Dimension(2, kernelSize, kernelSize),
                    weightCount,
                    biasCount,
                    new INeuronConfiguration[]
                    {
                        
                    }
                    )
                );

            var originalConvolutionCalculator = new NaiveConvolutionCalculator();

            var convolutionCalculator = new TestCSharpConvolutionCalculator(
                (k, d, t) =>
                {
                    if (!d.Check(
                        v => Math.Abs(v - neuronValue) < float.Epsilon
                        ))
                    {
                        throw new InternalTestFailureException("neuronValue");
                    }

                    if (!k.Check(
                        v => Math.Abs(v - weightValue) < float.Epsilon,
                        v => Math.Abs(v - biasValue) < float.Epsilon
                        ))
                    {
                        throw new InternalTestFailureException("weightValue || biasValue");
                    }

                    originalConvolutionCalculator.CalculateConvolutionWithOverwrite(
                        k,
                        d,
                        t
                        );
                },
                (k, d, t) =>
                {
                    if (!d.Check(
                        v => Math.Abs(v - neuronValue) < float.Epsilon
                        ))
                    {
                        throw new InternalTestFailureException("neuronValue");
                    }

                    if (!k.Check(
                        v => Math.Abs(v - weightValue) < float.Epsilon,
                        v => Math.Abs(v - biasValue) < float.Epsilon
                        ))
                    {
                        throw new InternalTestFailureException("weightValue || biasValue");
                    }

                    originalConvolutionCalculator.CalculateConvolutionWithIncrement(
                        k,
                        d,
                        t
                        );
                },
                null,
                null
                );

            var functionActivator = new TestFunctionActivator();

            var f = new CSharpFullConnected_ConvolutionLayerPropagator(
                previousLayerContainer,
                currentLayerContainer,
                convolutionCalculator,
                functionActivator
                );

            //fill the data

            previousLayerContainer.NetMem.Fill(neuronValue);
            previousLayerContainer.StateMem.Fill(neuronValue);
            previousLayerContainer.BiasMem.Clear();

            currentLayerContainer.NetMem.Clear();
            currentLayerContainer.StateMem.Clear();
            currentLayerContainer.BiasMem.Fill(biasValue);
            currentLayerContainer.WeightMem.Fill(weightValue);

            //calculate

            f.ComputeLayer();

            //check results
            Assert.IsTrue(currentLayerContainer.NetMem.All(j => Math.Abs(j - resultValue) < float.Epsilon));
            Assert.IsTrue(currentLayerContainer.StateMem.All(j => Math.Abs(j - resultValue) < float.Epsilon));

        }
    }

    public class TestCSharpConvolutionCalculator : ICSharpConvolutionCalculator
    {
        private readonly Action<IReferencedKernelBiasContainer, IReferencedSquareFloat, IReferencedSquareFloat> _forwardOverwrite;
        private readonly Action<IReferencedKernelBiasContainer, IReferencedSquareFloat, IReferencedSquareFloat> _forwardIncrement;
        private readonly Action<IReferencedSquareFloat, IReferencedSquareFloat, IReferencedSquareFloat> _backIncrement;
        private readonly Action<IReferencedSquareFloat, IReferencedSquareFloat, IReferencedSquareFloat> _backOverwrite;

        public TestCSharpConvolutionCalculator(
            Action<IReferencedKernelBiasContainer, IReferencedSquareFloat, IReferencedSquareFloat> forwardOverwrite,
            Action<IReferencedKernelBiasContainer, IReferencedSquareFloat, IReferencedSquareFloat> forwardIncrement,
            Action<IReferencedSquareFloat, IReferencedSquareFloat, IReferencedSquareFloat> backIncrement,
            Action<IReferencedSquareFloat, IReferencedSquareFloat, IReferencedSquareFloat> backOverwrite
            )
        {
            //any subset of parameters allowed to be null

            _forwardOverwrite = forwardOverwrite;
            _forwardIncrement = forwardIncrement;
            _backIncrement = backIncrement;
            _backOverwrite = backOverwrite;
        }

        public void CalculateBackConvolutionWithIncrement(
            IReferencedSquareFloat dzdy, 
            IReferencedSquareFloat dedz, 
            IReferencedSquareFloat target
            )
        {
            if (_backIncrement == null)
            {
                throw new InvalidOperationException();
            }

            _backIncrement(dzdy, dedz, target);
        }

        public void CalculateBackConvolutionWithOverwrite(
            IReferencedSquareFloat dzdy, 
            IReferencedSquareFloat dedz, 
            IReferencedSquareFloat target
            )
        {
            if (_backOverwrite == null)
            {
                throw new InvalidOperationException();
            }

            _backOverwrite(dzdy, dedz, target);
        }

        public void CalculateConvolutionWithOverwrite(
            IReferencedKernelBiasContainer kernelBiasContainer, 
            IReferencedSquareFloat dataToConvolute, 
            IReferencedSquareFloat target
            )
        {
            if (_forwardOverwrite == null)
            {
                throw new InvalidOperationException();
            }

            _forwardOverwrite(kernelBiasContainer, dataToConvolute, target);
        }

        public void CalculateConvolutionWithIncrement(
            IReferencedKernelBiasContainer kernelBiasContainer, 
            IReferencedSquareFloat dataToConvolute, 
            IReferencedSquareFloat target
            )
        {
            if (_forwardIncrement == null)
            {
                throw new InvalidOperationException();
            }

            _forwardIncrement(kernelBiasContainer, dataToConvolute, target);
        }
    }


    public class TestFunctionActivator : IFunctionActivator
    {
        public void Apply(
            IFunction activationFunction, 
            IReferencedSquareFloat currentNet, 
            IReferencedSquareFloat currentState
            )
        {
            if (activationFunction == null)
            {
                throw new ArgumentNullException("activationFunction");
            }
            if (currentNet == null)
            {
                throw new ArgumentNullException("currentNet");
            }
            if (currentState == null)
            {
                throw new ArgumentNullException("currentState");
            }

            //nothing to do except a copying

            for (var j = 0; j < currentNet.Height; j++)
            {
                for (var i = 0; i < currentNet.Width; i++)
                {
                    var v = currentNet.GetValueFromCoordSafely(i, j);

                    currentState.SetValueFromCoordSafely(i, j, v);
                }
            }
        }
    }

    public static class ReferencedHelper
    {
        public static bool Check(
            this IReferencedKernelBiasContainer kb,
            Func<float, bool> kernelPredicate,
            Func<float, bool> biasPredicate
            )
        {
            if (!biasPredicate(kb.Bias))
            {
                return false;
            }

            for (var w = 0; w < kb.Width; w++)
            {
                for (var h = 0; h < kb.Height; h++)
                {
                    var v = kb.GetValueFromCoordSafely(w, h);

                    if (!kernelPredicate(v))
                    {
                        return false;
                    }
                }
            }

            return true;
        }
        
        public static bool Check(
            this IReferencedSquareFloat s,
            Func<float, bool> kernelPredicate
            )
        {
            for (var w = 0; w < s.Width; w++)
            {
                for (var h = 0; h < s.Height; h++)
                {
                    var v = s.GetValueFromCoordSafely(w, h);

                    if (!kernelPredicate(v))
                    {
                        return false;
                    }
                }
            }

            return true;
        }
    }
}
