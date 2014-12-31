using System;
using System.Collections.Generic;
using MyNN.Common.NewData.DataSet;
using MyNN.Common.NewData.Item;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation1
{
    /// <summary>
    /// Correct but OBSOLETE OpenCL provider for generation #1 distance providers.
    /// </summary>
    public class OpenCLDistanceDictProvider : CLProvider
    {
        private readonly new List<IDataItem> _fxwList;

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

        public OpenCLDistanceDictProvider(
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

        public OpenCLDistanceDictProvider(List<IDataItem> fxwList)
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
                    FxwMem.Array[fxwIndex++] = i;
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
            FxwMem = this.CreateFloatMem(
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