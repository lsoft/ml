using System;
using System.Collections.Generic;
using System.Linq;
using OpenCL.Net.OpenCL.DeviceChooser;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace OpenCL.Net.OpenCL
{
    public class CLProvider : IDisposable
    {
        private readonly IDeviceChooser _deviceChooser;
        private Cl.Device _device;
        private Cl.Context _context;
        private Cl.CommandQueue _commandQueue;

        private readonly List<BaseMem> _mems;
        private readonly List<Kernel> _kernels;

        public Cl.DeviceType ChoosedDeviceType
        {
            get;
            private set;
        }

        public CLProvider(
            IDeviceChooser deviceChooser,
            bool silentStart)
        {
            if (deviceChooser == null)
            {
                throw new ArgumentNullException("deviceChooser");
            }

            _deviceChooser = deviceChooser;
            
            //ищем подходящее устройство opencl
            ChooseDevice();

            if (!silentStart)
            {
                //выводим его параметры
                GetDeviceInfo();
            }

            //создаем контекст вычислений
            CreateContext();

            //создаем очередь команд
            CreateCommandQueue();

            //создаем пустые структуры
            _mems = new List<BaseMem>();
            _kernels = new List<Kernel>();
        }

        public CLProvider(
            bool silentStart = true)
             : this(new IntelCPUDeviceChooser(), silentStart)
        {
        }

        #region opencl init

        private void CreateCommandQueue()
        {
            Cl.ErrorCode error;

            _commandQueue = Cl.CreateCommandQueue(_context, _device, (Cl.CommandQueueProperties)0, out error);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to CreateCommandQueue: {0}!", error));
            }
        }

        private void CreateContext()
        {
            Cl.ErrorCode error;

            _context = Cl.CreateContext(null, 1, new[] { _device }, null, IntPtr.Zero, out error);

            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidOperationException(string.Format("Unable to retrieve an OpenCL Context, error was: {0}!", error));
            }
        }

        public void GetDeviceInfo()
        {
            Cl.ErrorCode error;

            var name = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.Name, out error).ToString();
            var vendor = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.Vendor, out error).ToString();
            var openclVersion = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.Version, out error).ToString();
            var driverVersion = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.DriverVersion, out error).ToString();
            var globalMemory = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.GlobalMemSize, out error).CastTo<long>();
            var maxSamplers = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.MaxSamplers, out error).CastTo<int>();
            var extensions = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.Extensions, out error).ToString();
            var preferredFloat = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.PreferredVectorWidthFloat, out error).CastTo<int>();
            var preferredDouble = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.PreferredVectorWidthDouble, out error).CastTo<int>();
            var preferredShort = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.PreferredVectorWidthShort, out error).CastTo<int>();
            var preferredInt = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.PreferredVectorWidthInt, out error).CastTo<int>();
            var preferredLong = Cl.GetDeviceInfo(_device, Cl.DeviceInfo.PreferredVectorWidthLong, out error).CastTo<int>();

            Console.WriteLine("OpenCL device: " + name);
            Console.WriteLine("OpenCL vendor: " + vendor);
            Console.WriteLine("OpenCL version: " + openclVersion);
            Console.WriteLine("OpenCL driver version: " + driverVersion);
            Console.WriteLine(
                string.Format(
                    "Device global memory: {0} MB",
                    (int)(globalMemory / 1024 / 1024)));
            Console.WriteLine("Max samplers: " + maxSamplers);


            Console.WriteLine("Supported extensions: " + "\r\n    " + extensions.Replace(" ", "\r\n    "));

            if (extensions.Contains("cl_khr_fp64"))
            {
                Console.WriteLine("DOUBLES DOES SUPPORTED");
            }
            else
            {
                Console.WriteLine("DOUBLES DOES NOT SUPPORTED");
            }

            Console.WriteLine("Preferred vector width float: " + preferredFloat);
            Console.WriteLine("Preferred vector width double: " + preferredDouble);
            Console.WriteLine("Preferred vector width short: " + preferredShort);
            Console.WriteLine("Preferred vector width int: " + preferredInt);
            Console.WriteLine("Preferred vector width long: " + preferredLong);
        }

        private void ChooseDevice()
        {
            //Cl.ErrorCode error;

            //var platforms = Cl.GetPlatformIDs(out error);
            //if (error != Cl.ErrorCode.Success)
            //{
            //    throw new InvalidOperationException(string.Format("Unable to retrieve an OpenCL Device, error was: {0}!", error));
            //}

            //Cl.Device? device = null;

            //Cl.DeviceType[] devicePriority =
            //{
            //    //Cl.DeviceType.Cpu, //!!!
            //    Cl.DeviceType.Gpu, 
            //    Cl.DeviceType.Accelerator, 
            //    Cl.DeviceType.All, 
            //    Cl.DeviceType.Cpu
            //};

            //foreach (var deviceType in devicePriority)
            //{
            //    //look for GPUs first
            //    foreach (var platform in platforms)//.Skip(1))
            //    {
            //        var deviceIds = Cl.GetDeviceIDs(platform, deviceType, out error);
            //        if (deviceIds.Any())
            //        {
            //            device = deviceIds.First();
            //            this.ChoosedDeviceType = deviceType;
            //            break;
            //        }
            //    }

            //    if (device != null)
            //    {
            //        break;
            //    }
            //}

            //if (error != Cl.ErrorCode.Success)
            //{
            //    throw new InvalidOperationException(string.Format("Unable to retrieve an OpenCL Device, error was: {0}!", error));
            //}

            Cl.DeviceType choosedDeviceType;
            _deviceChooser.ChooseDevice(out choosedDeviceType, out _device);

            this.ChoosedDeviceType = choosedDeviceType;
        }

        #endregion

        /// <summary>
        /// загружаем программу и параметры
        /// </summary>
        /// <param name="source"></param>
        /// <param name="kernelName"></param>
        public Kernel CreateKernel(
            string source,
            string kernelName)
        {
            Cl.ErrorCode error;

            var program = Cl.CreateProgramWithSource(_context, 1, new[] { source }, null, out error);
            if (error != Cl.ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to run Cl.CreateProgramWithSource for program: {0}!", error));
            }

            error = Cl.BuildProgram(
                program,
                1,
                new[] { _device },
                string.Empty,
                //"-cl-opt-disable -cl-single-precision-constant -cl-fast-relaxed-math", 
                //"-cl-denorms-are-zero",
                //"-cl-single-precision-constant",
                null,
                IntPtr.Zero);
            if (error != Cl.ErrorCode.Success)
            {
                var infoBuffer = new Cl.InfoBuffer(new IntPtr(90000));
                IntPtr retSize;
                Cl.GetProgramBuildInfo(program, _device, Cl.ProgramBuildInfo.Log, new IntPtr(90000), infoBuffer, out retSize);

                throw new InvalidProgramException("Error building program:\n" + infoBuffer.ToString());
            }

            var buildStatus = Cl.GetProgramBuildInfo(program, _device, Cl.ProgramBuildInfo.Status, out error).CastTo<Cl.BuildStatus>();
            if (buildStatus != Cl.BuildStatus.Success)
            {
                throw new InvalidProgramException(string.Format("GetProgramBuildInfo returned {0} for program!", buildStatus));
            }

            var k = new Kernel(
                _commandQueue,
                program,
                kernelName);

            _kernels.Add(k);

            return k;
        }

        public MemUint CreateUintMem(
            int arrayLength,
            Cl.MemFlags flags)
        {
            var memi = new MemUint(
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memi);

            return memi;
        }

        public MemInt CreateIntMem(
            int arrayLength,
            Cl.MemFlags flags)
        {
            var memi = new MemInt(
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memi);

            return memi;
        }

        public MemFloat CreateFloatMem(
            int arrayLength,
            Cl.MemFlags flags)
        {
            var memf = new MemFloat(
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memf);

            return memf;
        }

        public MemDouble CreateDoubleMem(
            int arrayLength,
            Cl.MemFlags flags)
        {
            var memd = new MemDouble(
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memd);

            return memd;
        }

        public void QueueFinish()
        {
            _commandQueue.Finish();
        }

        public virtual void Dispose()
        {
            foreach (var m in _mems)
            {
                m.Dispose();
            }

            foreach (var k in _kernels)
            {
                k.Dispose();
            }

            Cl.ReleaseCommandQueue(_commandQueue);
            Cl.ReleaseContext(_context);
        }
    }
}
