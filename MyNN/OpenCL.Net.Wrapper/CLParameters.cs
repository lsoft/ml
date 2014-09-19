using System;


namespace OpenCL.Net.Wrapper
{
    public class CLParameters
    {
        public const int IntelVendorId = 32902;
        public const int NvidiaVendorId = 4318;
        public const int AMDVendorId = 4098;

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

        public int VendorId
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

        public ulong GlobalMemorySize
        {
            get;
            private set;
        }

        public ulong LocalMemorySize
        {
            get;
            private set;
        }

        public uint MaxSamplers
        {
            get;
            private set;
        }

        public uint NumComputeUnits
        {
            get;
            private set;
        }

        public uint MaxMemAllocSize
        {
            get;
            private set;
        }

        public uint MaxWorkGroupSize
        {
            get;
            private set;
        }

        public string Extensions
        {
            get;
            private set;
        }

        public uint PreferredFloat
        {
            get;
            private set;
        }


        public uint PreferredDouble
        {
            get;
            private set;
        }

        public uint PreferredShort
        {
            get;
            private set;
        }

        public uint PreferredInt
        {
            get;
            private set;
        }

        public uint PreferredLong
        {
            get;
            private set;
        }

        public bool IsImageSupport
        {
            get;
            private set;
        }

        public uint Image2DMaxWidth
        {
            get;
            private set;
        }

        public uint Image2DMaxHeight
        {
            get;
            private set;
        }
        
        public bool IsVendorIntel
        {
            get
            {
                return
                    VendorId == IntelVendorId;
            }
        }

        public bool IsVendorAMD
        {
            get
            {
                return
                    VendorId == AMDVendorId;
            }
        }

        public bool IsVendorNvidia
        {
            get
            {
                return
                    VendorId == NvidiaVendorId;
            }
        }

        public CLParameters(
            Device device)
        {
            ErrorCode error;

            DeviceName = Cl.GetDeviceInfo(device, DeviceInfo.Name, out error).ToString();
            Vendor = Cl.GetDeviceInfo(device, DeviceInfo.Vendor, out error).ToString();
            VendorId = Cl.GetDeviceInfo(device, DeviceInfo.VendorId, out error).CastTo<int>();
            OpenclVersion = Cl.GetDeviceInfo(device, DeviceInfo.Version, out error).ToString();
            DriverVersion = Cl.GetDeviceInfo(device, DeviceInfo.DriverVersion, out error).ToString();
            GlobalMemorySize = Cl.GetDeviceInfo(device, DeviceInfo.GlobalMemSize, out error).CastTo<ulong>();
            LocalMemorySize = Cl.GetDeviceInfo(device, DeviceInfo.LocalMemSize, out error).CastTo<ulong>();
            MaxSamplers = Cl.GetDeviceInfo(device, DeviceInfo.MaxSamplers, out error).CastTo<uint>();
            NumComputeUnits = Cl.GetDeviceInfo(device, DeviceInfo.MaxComputeUnits, out error).CastTo<uint>();
            MaxMemAllocSize = Cl.GetDeviceInfo(device, DeviceInfo.MaxMemAllocSize, out error).CastTo<uint>();
            MaxWorkGroupSize = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, out error).CastTo<uint>();

            Extensions = Cl.GetDeviceInfo(device, DeviceInfo.Extensions, out error).ToString();
            PreferredFloat = Cl.GetDeviceInfo(device, DeviceInfo.PreferredVectorWidthFloat, out error).CastTo<uint>();
            PreferredDouble = Cl.GetDeviceInfo(device, DeviceInfo.PreferredVectorWidthDouble, out error).CastTo<uint>();
            PreferredShort = Cl.GetDeviceInfo(device, DeviceInfo.PreferredVectorWidthShort, out error).CastTo<uint>();
            PreferredInt = Cl.GetDeviceInfo(device, DeviceInfo.PreferredVectorWidthInt, out error).CastTo<uint>();
            PreferredLong = Cl.GetDeviceInfo(device, DeviceInfo.PreferredVectorWidthLong, out error).CastTo<uint>();

            IsImageSupport = Cl.GetDeviceInfo(device, DeviceInfo.ImageSupport, out error).CastTo<bool>();
            Image2DMaxWidth = Cl.GetDeviceInfo(device, DeviceInfo.Image2DMaxWidth, out error).CastTo<uint>();
            Image2DMaxHeight = Cl.GetDeviceInfo(device, DeviceInfo.Image2DMaxHeight, out error).CastTo<uint>();
        }


        public void DumpToConsole()
        {
            Console.WriteLine("OpenCL device: " + DeviceName);
            Console.WriteLine("OpenCL vendor: [{0}] {1}", VendorId, Vendor);
            Console.WriteLine("OpenCL version:  " + OpenclVersion);
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

            Console.WriteLine("Image support: " + IsImageSupport);
            Console.WriteLine("Image2D max width: " + Image2DMaxWidth);
            Console.WriteLine("Image2D max height: " + Image2DMaxHeight);
        }
    }
}
