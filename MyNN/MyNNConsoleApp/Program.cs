using System;
using System.CodeDom;
using MathNet.Numerics.Integration.Algorithms;
using MyNN.Common.Other;
using MyNN.Common.Randomizer;
using MyNNConsoleApp.DBN;
using MyNNConsoleApp.RefactoredForDI;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using OpenCvSharp.CPlusPlus.Flann;

namespace MyNNConsoleApp
{
    class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            using (new CombinedConsole("console.log"))
            {
                CompareBP.DoCompare();
                //CompareBPGPU.DoCompare();


                Console.WriteLine(".......... press any key to exit");
                Console.ReadLine();
            }
        }
    }
}
