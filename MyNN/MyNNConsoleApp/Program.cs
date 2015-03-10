using System;
using MyNNConsoleApp.Conv;
using MyNNConsoleApp.DeDyRefactor;

namespace MyNNConsoleApp
{
    class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {
                //Convolute2.Do();

                //CSharpRefactorChecker.DoTrain();
                //CPURefactorChecker.DoTrain();
                //GPURefactorChecker.DoTrain();
                //GPUDropoutRefactorChecker.DoTrain();
                //GPUDropconnectRefactorChecker.DoTrain();

                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
