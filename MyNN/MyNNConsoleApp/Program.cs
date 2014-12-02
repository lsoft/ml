using System;
using MyNNConsoleApp.DBN;
using MyNNConsoleApp.RefactoredForDI;

namespace MyNNConsoleApp
{
    class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {
                TuneSDAE.Tune();


                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
