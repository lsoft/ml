using System;

namespace MyNN.OutputConsole
{
    public class DefaultOutputConsole : IOutputConsole
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
            Console.SetCursorPosition(0, Console.CursorTop);
        }
    }
}