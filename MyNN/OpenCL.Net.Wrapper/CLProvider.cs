using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using OpenCL.Net.Extensions;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace OpenCL.Net.Wrapper
{
    public class CLProvider : IDisposable
    {
        private readonly IDeviceChooser _deviceChooser;
        private Device _device;
        private Context _context;
        private CommandQueue _commandQueue;

        private readonly List<IMemWrapper> _mems;
        private readonly List<Kernel> _kernels;

        public Device ChoosedDevice
        {
            get
            {
                return
                    _device;
            }
        }

        public DeviceType ChoosedDeviceType
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
            _mems = new List<IMemWrapper>();
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
            ErrorCode error;

            _commandQueue = Cl.CreateCommandQueue(_context, _device, (CommandQueueProperties)0, out error);
            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to CreateCommandQueue: {0}!", error));
            }
        }

        private void CreateContext()
        {
            ErrorCode error;

            _context = Cl.CreateContext(null, 1, new[] { _device }, null, IntPtr.Zero, out error);

            if (error != ErrorCode.Success)
            {
                throw new InvalidOperationException(string.Format("Unable to retrieve an OpenCL Context, error was: {0}!", error));
            }
        }

        private void ChooseDevice()
        {
            //ErrorCode error;

            //var platforms = Cl.GetPlatformIDs(out error);
            //if (error != ErrorCode.Success)
            //{
            //    throw new InvalidOperationException(string.Format("Unable to retrieve an OpenCL Device, error was: {0}!", error));
            //}

            //Device? device = null;

            //DeviceType[] devicePriority =
            //{
            //    //DeviceType.Cpu, //!!!
            //    DeviceType.Gpu, 
            //    DeviceType.Accelerator, 
            //    DeviceType.All, 
            //    DeviceType.Cpu
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

            //if (error != ErrorCode.Success)
            //{
            //    throw new InvalidOperationException(string.Format("Unable to retrieve an OpenCL Device, error was: {0}!", error));
            //}

            DeviceType choosedDeviceType;
            _deviceChooser.ChooseDevice(out choosedDeviceType, out _device);

            this.ChoosedDeviceType = choosedDeviceType;
        }

        #endregion


        private string GetResourceFile(string resourceName)
        {
            var assembly = Assembly.GetExecutingAssembly();
            //var resourceName = "MyCompany.MyProduct.MyFile.txt";

            var names = assembly.GetManifestResourceNames().ToList();

            //names.ForEach(j => Console.WriteLine(j));

            var stream = assembly.GetManifestResourceStream(resourceName);

            if (stream != null)
            {
                try
                {
                    using (var reader = new StreamReader(stream))
                    {
                        var result = reader.ReadToEnd();
                        return result;
                    }
                }
                finally
                {
                    stream.Dispose();
                }
            }

            return null;

        }

        /// <summary>
        /// загружаем программу и параметры
        /// </summary>
        /// <param name="source">Текст кернела (или кернелов)</param>
        /// <param name="kernelName">Имя кернела</param>
        public Kernel CreateKernel(
            string source,
            string kernelName)
        {
            var fullTextStringBuilder = new StringBuilder();
            fullTextStringBuilder.Append(
                GetResourceFile("OpenCL.Net.Wrapper.KernelLibrary.WarpReduction.cl"));
            fullTextStringBuilder.Append(source);

            var fullText = fullTextStringBuilder.ToString();

            ErrorCode error;

            var program = Cl.CreateProgramWithSource(_context, 1, new[] { fullText }, null, out error);
            if (error != ErrorCode.Success)
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

            if (error != ErrorCode.Success)
            {
                var infoBuffer = new InfoBuffer(new IntPtr(90000));
                IntPtr retSize;
                Cl.GetProgramBuildInfo(program, _device, ProgramBuildInfo.Log, new IntPtr(90000), infoBuffer, out retSize);

                throw new InvalidProgramException("Error building program:\n" + infoBuffer.ToString());
            }

            var buildStatus = Cl.GetProgramBuildInfo(program, _device, ProgramBuildInfo.Status, out error).CastTo<BuildStatus>();
            if (buildStatus != BuildStatus.Success)
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
            MemFlags flags)
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
            MemFlags flags)
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
            MemFlags flags)
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
            MemFlags flags)
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
            MemFlags flags)
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
