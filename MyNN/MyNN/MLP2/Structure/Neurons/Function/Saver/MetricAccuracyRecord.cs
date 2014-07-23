using System;

namespace MyNN.MLP2.Saver
{
    public class MetricAccuracyRecord : IAccuracyRecord
    {
        private readonly int _epocheNumber;
        private readonly float _validationPerItemError;

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
    }
}