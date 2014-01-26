namespace MyNN.MLP2.Saver
{
    public class MLPAccuracyRecord : IAccuracyRecord
    {
        public float ValidationPerItemError
        {
            get;
            private set;
        }

        public int TotalCount
        {
            get;
            private set;
        }

        public int CorrectCount
        {
            get;
            private set;
        }

        public float CorrectPercent
        {
            get
            {
                return
                    ((int)(CorrectCount * 10000 / TotalCount)) / 100f;
            }
        }

        public MLPAccuracyRecord(
            int totalCount,
            int correctCount)
        {
            TotalCount = totalCount;
            CorrectCount = correctCount;
            ValidationPerItemError = float.MinValue;
        }

        public MLPAccuracyRecord(
            float validationPerItemError)
        {
            TotalCount = int.MinValue;
            CorrectCount = int.MinValue;
            ValidationPerItemError = validationPerItemError;
        }

        public MLPAccuracyRecord(float validationPerItemError, int totalCount, int correctCount)
        {
            ValidationPerItemError = validationPerItemError;
            TotalCount = totalCount;
            CorrectCount = correctCount;
        }
    }
}