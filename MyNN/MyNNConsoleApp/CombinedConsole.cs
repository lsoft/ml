using System;
using System.IO;

namespace MyNNConsoleApp
{
    internal class CombinedConsole : StreamWriter
    {
        private readonly TextWriter _originalConsole;
        private readonly TextWriter _fileConsole;

        public CombinedConsole(string filename)
            : base(Null.BaseStream)
        {
            this._originalConsole = Console.Out;
            this._fileConsole = new StreamWriter(new FileStream(filename, FileMode.Append, FileAccess.Write));

            Console.SetOut(this);
        }

        #region overrides

        public override void Write(bool value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(char value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(char[] buffer)
        {
            _originalConsole.Write(buffer);

            this._fileConsole.Write(buffer);
            this._fileConsole.Flush();
        }

        public override void Write(char[] buffer, int index, int count)
        {
            _originalConsole.Write(buffer, index, count);

            this._fileConsole.Write(buffer, index, count);
            this._fileConsole.Flush();
        }

        public override void Write(decimal value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(double value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(float value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(int value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(long value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(object value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(string format, object arg0)
        {
            _originalConsole.Write(format, arg0);

            this._fileConsole.Write(format, arg0);
            this._fileConsole.Flush();
        }

        public override void Write(string format, object arg0, object arg1)
        {
            _originalConsole.Write(format, arg0, arg1);

            this._fileConsole.Write(format, arg0, arg1);
            this._fileConsole.Flush();
        }

        public override void Write(string format, object arg0, object arg1, object arg2)
        {
            _originalConsole.Write(format, arg0, arg1, arg2);

            this._fileConsole.Write(format, arg0, arg1, arg2);
            this._fileConsole.Flush();
        }

        public override void Write(string format, params object[] arg)
        {
            _originalConsole.Write(format, arg);

            this._fileConsole.Write(format, arg);
            this._fileConsole.Flush();
        }

        public override void Write(uint value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(ulong value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void Write(string value)
        {
            _originalConsole.Write(value);

            this._fileConsole.Write(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(string value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine()
        {
            _originalConsole.WriteLine();

            this._fileConsole.WriteLine();
            this._fileConsole.Flush();
        }

        public override void WriteLine(bool value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(char value)
        {
            _originalConsole.WriteLine(value);
            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(char[] buffer)
        {
            _originalConsole.WriteLine(buffer);
            this._fileConsole.WriteLine(buffer);
            this._fileConsole.Flush();
        }

        public override void WriteLine(char[] buffer, int index, int count)
        {
            _originalConsole.WriteLine(buffer, index, count);
            this._fileConsole.WriteLine(buffer, index, count);
            this._fileConsole.Flush();
        }

        public override void WriteLine(decimal value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(double value)
        {
            _originalConsole.WriteLine(value);
            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(float value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(int value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(long value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(string format, object arg0)
        {
            _originalConsole.WriteLine(format, arg0);

            this._fileConsole.WriteLine(format, arg0);
            this._fileConsole.Flush();
        }

        public override void WriteLine(object value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(string format, object arg0, object arg1)
        {
            _originalConsole.WriteLine(format, arg0, arg1);

            this._fileConsole.WriteLine(format, arg0, arg1);
            this._fileConsole.Flush();
        }

        public override void WriteLine(string format, object arg0, object arg1, object arg2)
        {
            _originalConsole.WriteLine(format, arg0, arg1, arg2);

            this._fileConsole.WriteLine(format, arg0, arg1, arg2);
            this._fileConsole.Flush();
        }

        public override void WriteLine(string format, params object[] arg)
        {
            _originalConsole.WriteLine(format, arg);

            this._fileConsole.WriteLine(format, arg);
            this._fileConsole.Flush();
        }

        public override void WriteLine(uint value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        public override void WriteLine(ulong value)
        {
            _originalConsole.WriteLine(value);

            this._fileConsole.WriteLine(value);
            this._fileConsole.Flush();
        }

        #endregion

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                this.WriteLine("............... Close logging ...............");
                Console.SetOut(this._originalConsole);

                _fileConsole.Close();
                _fileConsole.Dispose();
            }

            base.Dispose(disposing);
        }
    }
}