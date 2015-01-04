using System;

namespace MyNNConsoleApp
{
    class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {
                


                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
