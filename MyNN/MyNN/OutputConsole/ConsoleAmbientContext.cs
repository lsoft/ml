using System;

namespace MyNN.OutputConsole
{
    public static class ConsoleAmbientContext
    {
        private static readonly object _lockObject = new object();

        private static IOutputConsole _console;
        public static IOutputConsole Console
        {
            get
            {
                lock (_lockObject)
                {
                    if (_console == null)
                    {
                        _console = new DefaultOutputConsole();
                    }

                    return _console;
                }
            }

            set
            {
                lock (_lockObject)
                {
                    if (value == null)
                    {
                        throw new ArgumentNullException("value");
                    }
                    if (_console != null)
                    {
                        throw new InvalidOperationException("_console != null");
                    }

                    _console = value;
                }
            }
        }

        public static void ResetConsole()
        {
            lock (_lockObject)
            {
                _console = null;
            }
        }
    }
}