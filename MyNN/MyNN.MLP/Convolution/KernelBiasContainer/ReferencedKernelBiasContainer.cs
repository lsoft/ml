using System;
using MyNN.MLP.Convolution.ReferencedSquareFloat;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Convolution.KernelBiasContainer
{
    public class ReferencedKernelBiasContainer : IReferencedKernelBiasContainer
    {
        private readonly float[] _biases;
        private readonly int _biasShift;

        public IDimension SpatialDimension
        {
            get;
            private set;
        }

        public int Width
        {
            get
            {
                return
                    this.SpatialDimension.Sizes[0];
            }
        }

        public int Height
        {
            get
            {
                return
                    this.SpatialDimension.Sizes[1];
            }
        }

        public IReferencedSquareFloat Kernel
        {
            get;
            private set;
        }

        public float Bias
        {
            get
            {
                return
                    _biases[_biasShift];
            }

            private set
            {
                _biases[_biasShift] = value;
            }
        }

        public ReferencedKernelBiasContainer(
            IDimension spatialDimension,
            float[] kernel,
            int kernelShift,
            float[] biases,
            int biasShift
            )
        {
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }
            if (kernel == null)
            {
                throw new ArgumentNullException("kernel");
            }
            if (biases == null)
            {
                throw new ArgumentNullException("biases");
            }
            if (spatialDimension.DimensionCount != 2)
            {
                throw new ArgumentException("spatialDimension.DimensionCount != 2");
            }

            SpatialDimension = spatialDimension;
            _biases = biases;
            _biasShift = biasShift;

            this.Kernel = new ReferencedSquareFloat.ReferencedSquareFloat(
                this.SpatialDimension,
                kernel,
                kernelShift
                );
        }

        public void ReadFrom(
            IReferencedKernelBiasContainer source
            )
        {
            if (source == null)
            {
                throw new ArgumentNullException("source");
            }
            if (source.Kernel.SpatialDimension.IsEqual(this.Kernel.SpatialDimension))
            {
                throw new ArgumentException("source.Kernel.SpatialDimension.IsEqual(this.Kernel.SpatialDimension)");
            }

            //копируем веса (кернел)
            for (var y = 0; y < this.Height; y++)
            {
                for (var x = 0; x < this.Width; x++)
                {
                    var v = source.GetValueFromCoordSafely(x, y);
                    source.Kernel.SetValueFromCoordSafely(x, y, v);
                }
            }

            //копируем биас
            this.Bias = source.Bias;
        }

        public float GetValueFromCoordSafely(int fromw, int fromh)
        {
            if (fromw < 0 || fromw >= Width)
            {
                throw new InvalidOperationException("fromw < 0 || fromw >= Width");
            }
            if (fromh < 0 || fromh >= Height)
            {
                throw new InvalidOperationException("fromh < 0 || fromh >= Height");
            }

            return
                this.Kernel.GetValueFromCoordSafely(fromw, fromh);
        }

        public void IncrementBy(
            IReferencedSquareFloat nablaContainer,
            float nablaBias,
            float batchSize
            )
        {
            if (nablaContainer == null)
            {
                throw new ArgumentNullException("nablaContainer");
            }
            if (!nablaContainer.SpatialDimension.IsEqual(this.Kernel.SpatialDimension))
            {
                throw new ArgumentException("nablaContainer.SpatialDimension.IsEqual(this.Kernel.SpatialDimension)");
            }

            //копируем веса (кернел)
            for (var y = 0; y < this.Height; y++)
            {
                for (var x = 0; x < this.Width; x++)
                {
                    var v = nablaContainer.GetValueFromCoordSafely(x, y);
                    v /= batchSize;
                    this.Kernel.AddValueFromCoordSafely(x, y, v);
                }
            }

            //копируем биас
            this.Bias += nablaBias / batchSize;
        }
    }
}