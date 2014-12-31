using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator
{
    public static class DoDfAmbientContext
    {
        private static readonly object _lockObject = new object();

        private static bool _disableExponential = false;
        public static bool DisableExponential
        {
            get
            {
                lock (_lockObject)
                {
                    return
                        _disableExponential;
                }
            }

            set
            {
                lock (_lockObject)
                {
                    _disableExponential = value;
                }
            }
        }

        public static void ResetConsole()
        {
            lock (_lockObject)
            {
                _disableExponential = false;
            }
        }
    }
}
