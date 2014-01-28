using System;
using System.Collections.Generic;
using MyNN.Data;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.DeviceChooser;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict
{
    public class DistanceDictCLProvider : CLProvider
    {
        private readonly List<DataItem> _fxwList;

        public MemInt IndexMem
        {
            get;
            private set;
        }

        public MemFloat FxwMem
        {
            get;
            private set;
        }

        public MemFloat DistanceMem
        {
            get;
            private set;
        }

        public DistanceDictCLProvider(
            IDeviceChooser deviceChooser, 
            bool silentStart, 
            List<DataItem> fxwList)
                : base(deviceChooser, silentStart)
        {
            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }

            _fxwList = fxwList;

            this.GenerateMems();
            this.FillMems();
            this.WriteMems();
        }

        public DistanceDictCLProvider(List<DataItem> fxwList)
            : this(
                new IntelCPUDeviceChooser(),
                true,
                fxwList)
        {
        }

        private void WriteMems()
        {
            FxwMem.Write(BlockModeEnum.Blocking);
            DistanceMem.Write(BlockModeEnum.Blocking);
            IndexMem.Write(BlockModeEnum.Blocking);
        }

        private void FillMems()
        {
            //FxwMem
            var fxwIndex = 0;
            foreach (var fxwi in _fxwList)
            {
                foreach (var i in fxwi.Input)
                {
                    FxwMem.Array[fxwIndex++] = i;
                }
            }

            //IndexMem
            for (var cc = 0; cc < _fxwList.Count; cc++) //!!! тут оптимизировать - чертовски неэффективно
            {
                IndexMem.Array[cc] = GetIndexCount(cc, _fxwList.Count);
            }
        }

        private void GenerateMems()
        {
            FxwMem = this.CreateFloatMem(
                _fxwList.Count * _fxwList[0].Input.Length,
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadOnly);

            IndexMem = this.CreateIntMem(
                _fxwList.Count,
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadOnly);

            DistanceMem = this.CreateFloatMem(
                GetTotalCount(_fxwList.Count),
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.WriteOnly);
        }

        public static int GetIndexCount(int index, int count)
        {
            var result = 0;

            for (var cc = 0; cc < index; cc++)
            {
                result += (count - cc);
            }

            return result;
        }

        public static int GetTotalCount(int count)
        {
            var result = 0;

            for (var cc = count; cc > 0; cc--)
            {
                result += cc;
            }

            return result;
        }
    }
}