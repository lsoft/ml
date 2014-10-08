using System;

namespace MyNN.MLP.AccuracyRecord
{
    public class ClassificationAccuracyRecord : IAccuracyRecord
    {
        private readonly int _epocheNumber;
        private readonly int _totalCount;
        private readonly int _correctCount;
        private readonly float _perItemError;

        public int CorrectCount
        {
            get
            {
                return
                    _correctCount;
            }
        }

        public float PerItemError
        {
            get
            {
                return
                    _perItemError;
            }
        }

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
            int correctCount,
            float perItemError
            )
        {
            _epocheNumber = epocheNumber;
            _totalCount = totalCount;
            _correctCount = correctCount;
            _perItemError = perItemError;
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

        public bool IsBetterThan(IAccuracyRecord accuracyRecord)
        {
            if (accuracyRecord == null)
            {
                throw new ArgumentNullException("accuracyRecord");
            }

            var c2 = accuracyRecord as ClassificationAccuracyRecord;

            if (c2 == null)
            {
                throw new ArgumentNullException("c2");
            }

            //количество правильных ответов важнее, чем ошибка, поэтому проверяем в первую очередь
            if (this._correctCount != c2._correctCount)
            {
                return
                    this._correctCount > c2._correctCount;
            }
            else
            {
                return
                    this._perItemError < c2._perItemError; //inverted operation because of comparison of _error_
            }
        }
    }
}