using System;
using MyNN.Common.OutputConsole;
using MyNN.MLP.Convolution.Calculator.CSharp;
using MyNN.MLP.Convolution.ErrorCalculator.CSharp;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.LearningConfig;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Classic.Backpropagation.EpocheTrainer.Classic.Conv.Kernel
{
    internal class OutputLayerKernel
    {
        private readonly ILayerConfiguration _layerConfiguration;
        private readonly ILearningAlgorithmConfig _config;
        private readonly IErrorCalculator _errorCalculator;

        public OutputLayerKernel(
            ILayerConfiguration layerConfiguration,
            ILearningAlgorithmConfig config,
            IErrorCalculator errorCalculator
            )
        {
            if (layerConfiguration == null)
            {
                throw new ArgumentNullException("layerConfiguration");
            }
            if (config == null)
            {
                throw new ArgumentNullException("config");
            }
            if (errorCalculator == null)
            {
                throw new ArgumentNullException("errorCalculator");
            }

            _layerConfiguration = layerConfiguration;
            _config = config;
            _errorCalculator = errorCalculator;

            ConsoleAmbientContext.Console.WriteWarning(
                "Этот обучатель используется только на этапе отладки, так как он допускает, что слой сверток - последний слой"
                );
        }

        public void CalculateOverwrite(
            IReferencedSquareFloat currentLayerNet,
            IReferencedSquareFloat currentLayerState,
            float[] desiredOutput,
            IReferencedSquareFloat previousLayerState,
            IReferencedSquareFloat nablaKernel,
            ref float nablaBias,
            float learningRate,
            IReferencedSquareFloat currentLayerDeDz
            )
        {
            if (currentLayerNet == null)
            {
                throw new ArgumentNullException("currentLayerNet");
            }
            if (currentLayerState == null)
            {
                throw new ArgumentNullException("currentLayerState");
            }
            if (desiredOutput == null)
            {
                throw new ArgumentNullException("desiredOutput");
            }
            if (previousLayerState == null)
            {
                throw new ArgumentNullException("previousLayerState");
            }
            if (nablaKernel == null)
            {
                throw new ArgumentNullException("nablaKernel");
            }
            if (currentLayerDeDz == null)
            {
                throw new ArgumentNullException("currentLayerDeDz");
            }

            //вычисляем значение ошибки (dE/dz)
            _errorCalculator.CalculateError(
                currentLayerNet,
                currentLayerState,
                desiredOutput,
                _config.TargetMetrics,
                _layerConfiguration.LayerActivationFunction,
                currentLayerDeDz
                );

            //считаем наблу
            for (var a = 0; a < nablaKernel.Width; a++)
            {
                for (var b = 0; b < nablaKernel.Height; b++)
                {
                    var dEdw_ab = 0f; //kernel
                    var dEdb_ab = 0f; //bias

                    for (var i = 0; i < currentLayerNet.Width; i++)
                    {
                        for (var j = 0; j < currentLayerNet.Height; j++)
                        {
                            var dEdz_ij = currentLayerDeDz.GetValueFromCoordSafely(i, j);

                            var y = previousLayerState.GetValueFromCoordSafely(
                                i + a,
                                j + b
                                );

                            var w_mul = dEdz_ij * y * learningRate;
                            dEdw_ab += w_mul;

                            var b_mul = dEdz_ij * learningRate;
                            dEdb_ab += b_mul;

                        }
                    }

                    //вычислено dEdw_ab, dEdb_ab
                    nablaKernel.SetValueFromCoordSafely(a, b, dEdw_ab);
                    nablaBias = dEdb_ab * learningRate;
                }
            }
        }

        public void CalculateIncrement(
            IReferencedSquareFloat currentLayerNet,
            IReferencedSquareFloat currentLayerState,
            float[] desiredOutput,
            IReferencedSquareFloat previousLayerState,
            IReferencedSquareFloat nablaKernel,
            ref float nablaBias,
            float learningRate,
            IReferencedSquareFloat currentLayerDeDz
            )
        {
            if (currentLayerNet == null)
            {
                throw new ArgumentNullException("currentLayerNet");
            }
            if (currentLayerState == null)
            {
                throw new ArgumentNullException("currentLayerState");
            }
            if (desiredOutput == null)
            {
                throw new ArgumentNullException("desiredOutput");
            }
            if (previousLayerState == null)
            {
                throw new ArgumentNullException("previousLayerState");
            }
            if (nablaKernel == null)
            {
                throw new ArgumentNullException("nablaKernel");
            }
            if (currentLayerDeDz == null)
            {
                throw new ArgumentNullException("currentLayerDeDz");
            }

            //вычисляем значение ошибки (dE/dz)
            _errorCalculator.CalculateError(
                currentLayerNet,
                currentLayerState,
                desiredOutput,
                _config.TargetMetrics,
                _layerConfiguration.LayerActivationFunction,
                currentLayerDeDz
                );

            //считаем наблу
            for (var a = 0; a < nablaKernel.Width; a++)
            {
                for (var b = 0; b < nablaKernel.Height; b++)
                {
                    var dEdw_ab = 0f; //kernel
                    var dEdb_ab = 0f; //bias

                    for (var i = 0; i < currentLayerNet.Width; i++)
                    {
                        for (var j = 0; j < currentLayerNet.Height; j++)
                        {
                            var dEdz_ij = currentLayerDeDz.GetValueFromCoordSafely(i, j);

                            var y = previousLayerState.GetValueFromCoordSafely(
                                i + a,
                                j + b
                                );

                            var w_mul = dEdz_ij * y * learningRate;
                            dEdw_ab += w_mul;

                            var b_mul = dEdz_ij * learningRate;
                            dEdb_ab += b_mul;

                        }
                    }

                    //вычислено dEdw_ab, dEdb_ab
                    nablaKernel.AddValueFromCoordSafely(a, b, dEdw_ab);
                    nablaBias += dEdb_ab * learningRate;
                }
            }
        }
    }
}
