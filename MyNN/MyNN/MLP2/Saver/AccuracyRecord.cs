namespace MyNN.MLP2.Saver
{
    public class AccuracyRecord : IAccuracyRecord
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

        public AccuracyRecord(
            int totalCount,
            int correctCount)
        {
            TotalCount = totalCount;
            CorrectCount = correctCount;
            ValidationPerItemError = float.MinValue;
        }

        public AccuracyRecord(
            float validationPerItemError)
        {
            TotalCount = int.MinValue;
            CorrectCount = int.MinValue;
            ValidationPerItemError = validationPerItemError;
        }

        public AccuracyRecord(float validationPerItemError, int totalCount, int correctCount)
        {
            ValidationPerItemError = validationPerItemError;
            TotalCount = totalCount;
            CorrectCount = correctCount;
        }
    }
}