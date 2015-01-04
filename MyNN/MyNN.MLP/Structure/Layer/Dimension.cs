using System.Linq;

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

        public int TotalNeuronCount
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
                    ":",
                    this.Sizes);
        }


    }
}