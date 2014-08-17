namespace MyNN
{
    // ReSharper disable once InconsistentNaming
    public class diapint
    {
        public int Min
        {
            get;
            private set;
        }

        public int Max
        {
            get;
            private set;
        }

        public diapint(int min, int max)
        {
            Min = min;
            Max = max;

        }

        public bool IsValueInclusive(float testValue)
        {
            return
                testValue >= Min
                && testValue <= Max;
        }

        public override string ToString()
        {
            var result = string.Format(
                "[{0} <-> {1}]",
                Min,
                Max
                );

            return result;
        }
    }
}