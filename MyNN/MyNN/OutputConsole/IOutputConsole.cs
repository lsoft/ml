﻿using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MyNN.OutputConsole
{
    public interface IOutputConsole
    {
        void Write(string message);

        void Write(string message, params object[] p);

        void WriteWarning(string message);

        void WriteWarning(string message, params object[] p);

        void WriteLine(string message);

        void WriteLine(string message, params object[] p);

        void ReturnCarriage();
    }
}
