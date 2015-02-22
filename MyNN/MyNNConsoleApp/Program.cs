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
                //Convolute2.Do();


                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
