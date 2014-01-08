using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.OutputConsole;

namespace MyNN.Tests
{
    internal class TestOutputConsole : IOutputConsole
    {
        public void Write(string message)
        {
            Console.Write(message);
        }

        public void Write(string message, params object[] p)
        {
            Console.Write(message, p);
        }

        public void WriteLine(string message)
        {
            Console.WriteLine(message);
        }

        public void WriteLine(string message, params object[] p)
        {
            Console.WriteLine(message, p);
        }

        public void ReturnCarriage()
        {
            //nothing to do in test environment
        }
    }
}
