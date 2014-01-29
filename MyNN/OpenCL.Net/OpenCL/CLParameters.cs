using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL
{
    public class CLParameters
    {
        public string DeviceName
        {
            get;
            private set;
        }

        public string Vendor
        {
            get;
            private set;
        }

        public string OpenclVersion
        {
            get;
            private set;
        }

        public string DriverVersion
        {
            get;
            private set;
        }

        public long GlobalMemorySize
        {
            get;
            private set;
        }

        public long LocalMemorySize
        {
            get;
            private set;
        }

        public int MaxSamplers
        {
            get;
            private set;
        }

        public int NumComputeUnits
        {
            get;
            private set;
        }

        public int MaxMemAllocSize
        {
            get;
            private set;
        }

        public int MaxWorkGroupSize
        {
            get;
            private set;
        }


        public string Extensions
        {
            get;
            private set;
        }

        public int PreferredFloat
        {
            get;
            private set;
        }


        public int PreferredDouble
        {
            get;
            private set;
        }

        public int PreferredShort
        {
            get;
            private set;
        }

        public int PreferredInt
        {
            get;
            private set;
        }

        public int PreferredLong
        {
            get;
            private set;
        }

        public CLParameters(
            Cl.Device device)
        {
            Cl.ErrorCode error;

            DeviceName = Cl.GetDeviceInfo(device, Cl.DeviceInfo.Name, out error).ToString();
            Vendor = Cl.GetDeviceInfo(device, Cl.DeviceInfo.Vendor, out error).ToString();
            OpenclVersion = Cl.GetDeviceInfo(device, Cl.DeviceInfo.Version, out error).ToString();
            DriverVersion = Cl.GetDeviceInfo(device, Cl.DeviceInfo.DriverVersion, out error).ToString();
            GlobalMemorySize = Cl.GetDeviceInfo(device, Cl.DeviceInfo.GlobalMemSize, out error).CastTo<long>();
            LocalMemorySize = Cl.GetDeviceInfo(device, Cl.DeviceInfo.LocalMemSize, out error).CastTo<long>();
            MaxSamplers = Cl.GetDeviceInfo(device, Cl.DeviceInfo.MaxSamplers, out error).CastTo<int>();
            NumComputeUnits = Cl.GetDeviceInfo(device, Cl.DeviceInfo.MaxComputeUnits, out error).CastTo<int>();
            MaxMemAllocSize = Cl.GetDeviceInfo(device, Cl.DeviceInfo.MaxMemAllocSize, out error).CastTo<int>();
            MaxWorkGroupSize = Cl.GetDeviceInfo(device, Cl.DeviceInfo.MaxWorkGroupSize, out error).CastTo<int>();

            Extensions = Cl.GetDeviceInfo(device, Cl.DeviceInfo.Extensions, out error).ToString();
            PreferredFloat = Cl.GetDeviceInfo(device, Cl.DeviceInfo.PreferredVectorWidthFloat, out error).CastTo<int>();
            PreferredDouble = Cl.GetDeviceInfo(device, Cl.DeviceInfo.PreferredVectorWidthDouble, out error).CastTo<int>();
            PreferredShort = Cl.GetDeviceInfo(device, Cl.DeviceInfo.PreferredVectorWidthShort, out error).CastTo<int>();
            PreferredInt = Cl.GetDeviceInfo(device, Cl.DeviceInfo.PreferredVectorWidthInt, out error).CastTo<int>();
            PreferredLong = Cl.GetDeviceInfo(device, Cl.DeviceInfo.PreferredVectorWidthLong, out error).CastTo<int>();
        }


        public void DumpToConsole()
        {
            Console.WriteLine("OpenCL device: " + DeviceName);
            Console.WriteLine("OpenCL vendor: " + Vendor);
            Console.WriteLine("OpenCL version: " + OpenclVersion);
            Console.WriteLine("OpenCL driver version: " + DriverVersion);
            Console.WriteLine("Device global memory: {0} MB", (int) (GlobalMemorySize/1024/1024));
            Console.WriteLine("Device local memory: {0} KB", (int)(LocalMemorySize / 1024));
            Console.WriteLine("Max memory allocation size: {0} MB", (int)(MaxMemAllocSize / 1024 / 1024));
            Console.WriteLine("Max samplers: " + MaxSamplers);
            Console.WriteLine("Num of compute unit: " + NumComputeUnits);
            Console.WriteLine("Max work group size: " + MaxWorkGroupSize);


            Console.WriteLine("Supported extensions: " + "\r\n    " + Extensions.Replace(" ", "\r\n    "));

            if (Extensions.Contains("cl_khr_fp16"))
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("HALVES DOES SUPPORTED");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("HALVES DOES NOT SUPPORTED");
                Console.ResetColor();
            }

            if (Extensions.Contains("cl_khr_fp64"))
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("DOUBLES DOES SUPPORTED");
                Console.ResetColor();
            }
            else
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("DOUBLES DOES NOT SUPPORTED");
                Console.ResetColor();
            }

            Console.WriteLine("Preferred vector width float: " + PreferredFloat);
            Console.WriteLine("Preferred vector width double: " + PreferredDouble);
            Console.WriteLine("Preferred vector width short: " + PreferredShort);
            Console.WriteLine("Preferred vector width int: " + PreferredInt);
            Console.WriteLine("Preferred vector width long: " + PreferredLong);
        }
    }
}
