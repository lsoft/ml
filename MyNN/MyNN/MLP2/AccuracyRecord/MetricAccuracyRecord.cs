using System;

namespace MyNN.MLP2.AccuracyRecord
{
    public class MetricAccuracyRecord : IAccuracyRecord
    {
        private readonly int _epocheNumber;
        private readonly float _validationPerItemError;

        public float PerItemError
        {
            get
            {
                return
                    _validationPerItemError;
            }
        }

        public MetricAccuracyRecord(
            int epocheNumber,
            float validationPerItemError)
        {
            _epocheNumber = epocheNumber;
            _validationPerItemError = validationPerItemError;
        }

        public string GetTextResults()
        {
            var result = string.Format(
                "{0}: validation per-item error = {1}",
                DateTime.Now.ToString("yyyy.MM.dd HH:mm:ss"),
                _validationPerItemError);

            return
                result;
        }

        public bool IsBetterThan(IAccuracyRecord accuracyRecord)
        {
            if (accuracyRecord == null)
            {
                throw new ArgumentNullException("accuracyRecord");
            }

            var c2 = accuracyRecord as MetricAccuracyRecord;

            if (c2 == null)
            {
                throw new ArgumentNullException("c2");
            }

            return
                this._validationPerItemError < c2._validationPerItemError;
        }
    }
}