using System;
using MyNN.Common.Other;
using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Convolution.ReferencedSquareFloat
{
    public class ReferencedSquareFloat : IReferencedSquareFloat
    {
        private readonly float[] _array;
        private readonly int _shift;

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

        public ReferencedSquareFloat(
            IDimension spatialDimension,
            float[] array,
            int shift
            )
        {
            if (spatialDimension == null)
            {
                throw new ArgumentNullException("spatialDimension");
            }
            if (array == null)
            {
                throw new ArgumentNullException("array");
            }
            if (spatialDimension.DimensionCount != 2)
            {
                throw new ArgumentException("spatialDimension.DimensionCount != 2");
            }

            SpatialDimension = spatialDimension;

            _array = array;
            _shift = shift;
        }

        public void ReadFrom(float[] source)
        {
            if (source == null)
            {
                throw new ArgumentNullException("source");
            }
            if (source.Length != this.SpatialDimension.Multiplied)
            {
                throw new ArgumentException("source.Length != this.SpatialDimension.Multiplied");
            }

            source.CopyTo(_array, _shift);
        }

        public void AddValueFromCoordSafely(int readw, int readh, float value)
        {
            if (readw < 0 || readw >= Width)
            {
                throw new InvalidOperationException("readw < 0 || readw >= Width");
            }
            if (readh < 0 || readh >= Height)
            {
                throw new InvalidOperationException("readh < 0 || readh >= Height");
            }

            _array[_shift + (readh * Width + readw)] += value;
        }

        public void SetValueFromCoordSafely(int readw, int readh, float value)
        {
            if (readw < 0 || readw >= Width)
            {
                throw new InvalidOperationException("readw < 0 || readw >= Width");
            }
            if (readh < 0 || readh >= Height)
            {
                throw new InvalidOperationException("readh < 0 || readh >= Height");
            }

            _array[_shift + (readh * Width + readw)] = value;
        }

        public void ChangeValueFromCoordSafely(int readw, int readh, float value, int invert_overwrite)
        {
            if (readw < 0 || readw >= Width)
            {
                throw new InvalidOperationException("readw < 0 || readw >= Width");
            }
            if (readh < 0 || readh >= Height)
            {
                throw new InvalidOperationException("readh < 0 || readh >= Height");
            }

            var index = _shift + (readh*Width + readw);
            var prev = _array[index];

            _array[index] = value + prev * invert_overwrite;
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
                _array[_shift + (fromh * Width + fromw)];
        }

        public float GetValueFromCoordPaddedWithZeroes(int fromw, int fromh)
        {
            if (fromw < 0 || fromw >= Width)
            {
                return 0f;
            }
            if (fromh < 0 || fromh >= Height)
            {
                return 0f;
            }

            return
                _array[_shift + (fromh * Width + fromw)];
        }

        public void Clear()
        {
            _array.Clear(_shift, this.SpatialDimension.Multiplied);
        }
    }
}