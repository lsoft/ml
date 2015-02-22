using System.Linq;
using MyNN.Common.Other;

namespace MyNN.MLP.Structure.Layer
{
    public class Dimension : IDimension
    {
        public int DimensionCount
        {
            get;
            private set;
        }

        public int[] Sizes
        {
            get;
            private set;
        }

        public int Width
        {
            get
            {
                return
                    Sizes[0];
            }
        }

        public int Height
        {
            get
            {
                return
                    Sizes[1];
            }
        }

        public int Multiplied
        {
            get
            {
                return
                    Sizes.Aggregate(
                        1,
                        (current, i) => current * i);
            }
        }

        public Dimension(
            int dimensionCount, 
            params int[] sizes
            )
        {
            DimensionCount = dimensionCount;
            Sizes = sizes;
        }

        public string GetDimensionInformation(
            )
        {
            return
                string.Join(
                    "x",
                    this.Sizes);
        }

        public bool IsEqual(
            IDimension dim
            )
        {
            if (dim == null)
            {
                return false;
            }

            if(this.DimensionCount  != dim.DimensionCount)
            {
                return false;
            }

            if (!ArrayOperations.ValuesAreEqual(this.Sizes, dim.Sizes))
            {
                return false;
            }

            return true;
        }

        public IDimension Rescale(float scaleFactor)
        {
            return 
                new Dimension(
                    this.DimensionCount,
                    this.Sizes.ConvertAll(j => (int)(j * scaleFactor))
                    );
        }
    }
}