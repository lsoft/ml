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

        public Cl.Device ChoosedDevice
        {
            get
            {
                return
                    _device;
            }
        }

        public Cl.DeviceType ChoosedDeviceType
        {
            get;
            private set;
        }

        public CLParameters Parameters
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

            this.Parameters = new CLParameters(this.ChoosedDevice);

            if (!silentStart)
            {
                //выводим его параметры
                this.Parameters.DumpToConsole();
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
                //"-cl-mad-enable",
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

        public MemHalf CreateHalfMem(
            int arrayLength,
            Cl.MemFlags flags)
        {
            var memh = new MemHalf(
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memh);

            return memh;
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
