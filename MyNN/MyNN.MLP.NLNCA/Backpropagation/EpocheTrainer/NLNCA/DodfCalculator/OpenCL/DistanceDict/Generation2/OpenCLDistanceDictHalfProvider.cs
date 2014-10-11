using System;
using System.Collections.Generic;
using MyNN.Common.Data;
using MyNN.Common.Data.Set;
using MyNN.Common.Data.Set.Item;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation2
{
    /// <summary>
    /// Correct but OBSOLETE OpenCL provider for generation #2 distance providers that consumes HALF input representations.
    /// </summary>
    public class OpenCLDistanceDictHalfProvider : CLProvider
    {
        private readonly new List<IDataItem> _fxwList;

        public MemInt IndexMem
        {
            get;
            private set;
        }

        public MemHalf FxwMem
        {
            get;
            private set;
        }

        public MemFloat DistanceMem
        {
            get;
            private set;
        }

        public OpenCLDistanceDictHalfProvider(
            IDeviceChooser deviceChooser, 
            bool silentStart, 
            List<IDataItem> fxwList)
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

        public OpenCLDistanceDictHalfProvider(List<IDataItem> fxwList)
            : this(
                new IntelCPUDeviceChooser(false),
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
                    FxwMem.Array[fxwIndex++] = new System.Half(i);
                }
            }

            //IndexMem
            var accum = 0;
            var startCount = _fxwList.Count;
            for (var cc = 0; cc < _fxwList.Count; cc++)
            {
                IndexMem.Array[cc] = accum;
                accum += startCount;
                startCount--;
            }
        }

        private void GenerateMems()
        {
            FxwMem = this.CreateHalfMem(
                _fxwList.Count * _fxwList[0].Input.Length,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            IndexMem = this.CreateIntMem(
                _fxwList.Count,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            var totalCount = (_fxwList.Count + 1) * _fxwList.Count / 2;

            DistanceMem = this.CreateFloatMem(
                totalCount,
                MemFlags.CopyHostPtr | MemFlags.WriteOnly);
        }
    }
}