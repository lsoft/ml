using System;

namespace MyNN.MLP2.Saver
{
    public class ClassificationAccuracyRecord : IAccuracyRecord
    {
        private readonly int _epocheNumber;
        private readonly int _totalCount;
        private readonly int _correctCount;

        private float _correctPercent
        {
            get
            {
                return
                    ((int)(_correctCount * 10000 / _totalCount)) / 100f;
            }
        }

        public ClassificationAccuracyRecord(
            int epocheNumber,
            int totalCount,
            int correctCount)
        {
            _epocheNumber = epocheNumber;
            _totalCount = totalCount;
            _correctCount = correctCount;
        }

        public string GetTextResults()
        {
            var result = string.Format(
                "{0}: {1} correct out of {2}, {3}%",
                DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"),
                _correctCount,
                _totalCount,
                _correctPercent);

            return
                result;
        }
    }
}