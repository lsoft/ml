using System.Diagnostics;

namespace MyNN.Tests.MLP2.Sort
{
    [DebuggerDisplay("A = {AIndex}, B = {BIndex}, D = {Distance}")]
    internal class SortItem
    {
        public uint AIndex
        {
            get;
            private set;
        }

        public uint BIndex
        {
            get;
            private set;
        }

        public float Distance
        {
            get;
            private set;
        }

        public SortItem(uint aIndex, uint bIndex, float distance)
        {
            AIndex = aIndex;
            BIndex = bIndex;
            Distance = distance;
        }
    }
}