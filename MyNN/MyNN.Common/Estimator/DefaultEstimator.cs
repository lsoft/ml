using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.Common.Estimator
{

    public class DefaultEstimator
    {
        private readonly int _total;
        
        private DateTime _startTime;
        private DateTime _time;

        public DefaultEstimator(
            int total)
        {
            _total = total;
        }

        public void Start()
        {
            var now = DateTime.Now;

            _startTime = now;

            _time = now;
        }

        public TimeSpan Tick(int currentValue)
        {
            if (currentValue > _total)
            {
                throw new ArgumentException("currentValue > _total");
            }

            var now = DateTime.Now;

            _time = now;

            var taken = _time - _startTime;
            
            var totalTime = (long)(_total / (double) currentValue * taken.Ticks);
            var est = new TimeSpan(totalTime - taken.Ticks);

            return est;
        }
    }
}
