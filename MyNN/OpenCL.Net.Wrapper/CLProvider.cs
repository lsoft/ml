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
using OpenCL.Net.Wrapper.Resource;

namespace OpenCL.Net.Wrapper
{
    public class CLProvider : IDisposable
    {
        private readonly Device _device;
        private readonly Context _context;
        private readonly CommandQueue _commandQueue;

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
            
            //ищем подходящее устройство opencl
            DeviceType choosedDeviceType;
            deviceChooser.ChooseDevice(out choosedDeviceType, out _device);

            this.ChoosedDeviceType = choosedDeviceType;

            this.Parameters = new CLParameters(this.ChoosedDevice);

            if (!silentStart)
            {
                //выводим его параметры
                this.Parameters.DumpToConsole();
            }

            //создаем контекст вычислений
            _context = CreateContext();

            //создаем очередь команд
            _commandQueue = CreateCommandQueue();

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

        private CommandQueue CreateCommandQueue()
        {
            ErrorCode error;

            var commandQueue = Cl.CreateCommandQueue(_context, _device, (CommandQueueProperties)0, out error);
            if (error != ErrorCode.Success)
            {
                throw new InvalidProgramException(string.Format("Unable to CreateCommandQueue: {0}!", error));
            }

            return commandQueue;
        }

        private Context CreateContext()
        {
            ErrorCode error;

            var context = Cl.CreateContext(null, 1, new[] { _device }, null, IntPtr.Zero, out error);

            if (error != ErrorCode.Success)
            {
                throw new InvalidOperationException(string.Format("Unable to retrieve an OpenCL Context, error was: {0}!", error));
            }

            return
                context;
        }

        #endregion

        /// <summary>
        /// загружаем программу и параметры
        /// </summary>
        /// <param name="source">Текст кернела (или кернелов)</param>
        /// <param name="kernelName">Имя кернела</param>
        public Kernel CreateKernel(
            string source,
            string kernelName)
        {
            var err = new EmbeddedResourceReader();

            var fullTextStringBuilder = new StringBuilder();
            fullTextStringBuilder.Append(err.GetTextResourceFile("OpenCL.Net.Wrapper.KernelLibrary.Reduction.cl"));
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
            long arrayLength,
            MemFlags flags)
        {
            if (arrayLength <= 0L)
            {
                throw new ArgumentOutOfRangeException("arrayLength");
            }

            var memi = new MemUint(
                MemDisposed,
                _commandQueue,
                _context,
                (ulong)arrayLength,
                flags);

            this._mems.Add(memi);

            return memi;
        }

        public MemUint CreateUintMem(
            ulong arrayLength,
            MemFlags flags)
        {
            var memi = new MemUint(
                MemDisposed,
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memi);

            return memi;
        }

        public MemInt CreateIntMem(
            long arrayLength,
            MemFlags flags)
        {
            if (arrayLength <= 0L)
            {
                throw new ArgumentOutOfRangeException("arrayLength");
            }

            var memi = new MemInt(
                MemDisposed,
                _commandQueue,
                _context,
                (ulong)arrayLength,
                flags);

            this._mems.Add(memi);

            return memi;
        }


        public MemInt CreateIntMem(
            ulong arrayLength,
            MemFlags flags)
        {
            var memi = new MemInt(
                MemDisposed,
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memi);

            return memi;
        }

        public MemHalf CreateHalfMem(
            long arrayLength,
            MemFlags flags)
        {
            if (arrayLength <= 0L)
            {
                throw new ArgumentOutOfRangeException("arrayLength");
            }

            var memh = new MemHalf(
                MemDisposed,
                _commandQueue,
                _context,
                (ulong)arrayLength,
                flags);

            this._mems.Add(memh);

            return memh;
        }

        public MemHalf CreateHalfMem(
            ulong arrayLength,
            MemFlags flags)
        {
            var memh = new MemHalf(
                MemDisposed,
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memh);

            return memh;
        }


        public MemByte CreateByteMem(
            ulong arrayLength,
            MemFlags flags)
        {
            var memb = new MemByte(
                MemDisposed,
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memb);

            return memb;
        }

        public MemFloat CreateFloatMem(
            long arrayLength,
            MemFlags flags)
        {
            if (arrayLength <= 0L)
            {
                throw new ArgumentOutOfRangeException("arrayLength");
            }

            var memf = new MemFloat(
                MemDisposed,
                _commandQueue,
                _context,
                (ulong)arrayLength,
                flags);

            this._mems.Add(memf);

            return memf;
        }

        public MemFloat CreateFloatMem(
            ulong arrayLength,
            MemFlags flags)
        {
            var memf = new MemFloat(
                MemDisposed,
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memf);

            return memf;
        }

        public MemDouble CreateDoubleMem(
            ulong arrayLength,
            MemFlags flags)
        {
            var memd = new MemDouble(
                MemDisposed,
                _commandQueue,
                _context,
                arrayLength,
                flags);

            this._mems.Add(memd);

            return memd;
        }

        private void MemDisposed(Guid memGuid)
        {
            var removedCount = _mems.RemoveAll(j => j.MemGuid == memGuid);

            if (removedCount != 1)
            {
                throw new InvalidOperationException("Должен был удалиться 1 мем");
            }
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
