﻿using System;
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
                //GPURefactorChecker.DoTrain();
                //CPURefactorChecker.DoTrain();
                GPUDropoutRefactorChecker.DoTrain();

                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
