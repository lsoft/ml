using System;

namespace MyNNConsoleApp
{
    class Program
    {
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {

                TrainSDAE.DoTrain();

            }

            Console.WriteLine(".......... press any key to exit");
            Console.ReadLine();
        }
    }
}
