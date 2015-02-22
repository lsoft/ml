using MyNN.MLP.Structure.Layer;

namespace MyNN.MLP.Convolution.ReferencedSquareFloat
{
    public interface IReferencedSquareFloat
    {
        IDimension SpatialDimension
        {
            get;
        }

        int Width
        {
            get;
        }

        int Height
        {
            get;
        }

        void ReadFrom(float[] source);

        void AddValueFromCoordSafely(int readw, int readh, float value);

        void SetValueFromCoordSafely(int readw, int readh, float value);

        float GetValueFromCoordSafely(int fromw, int fromh);

        float GetValueFromCoordPaddedWithZeroes(int fromw, int fromh);

        void Clear();
    }
}