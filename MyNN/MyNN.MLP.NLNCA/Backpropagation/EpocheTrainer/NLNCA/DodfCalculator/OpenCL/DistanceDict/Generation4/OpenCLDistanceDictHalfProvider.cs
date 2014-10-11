using System;
using System.Collections.Generic;
using MyNN.Common.Data;
using MyNN.Common.Other;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;

namespace MyNN.MLP.NLNCA.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4
{
    /// <summary>
    /// Correct OpenCL provider for generation #4 distance providers that consumes HALF input representations.
    /// </summary>
    public class OpenCLDistanceDictHalfProvider : CLProvider
    {
        private readonly new List<IDataItem> _fxwList;
        private readonly uint _distanceMemElementCount;

        public MemUint IndexMem
        {
            get;
            private set;
        }

        public MemHalf FxwMem
        {
            get;
            private set;
        }

        public MemByte DistanceMem
        {
            get;
            private set;
        }

        public MemByte AccumMem
        {
            get;
            private set;
        }

        public ulong AccumulatorActualItemCount
        {
            get;
            private set;
        }

        public OpenCLDistanceDictHalfProvider(
            IDeviceChooser deviceChooser,
            bool silentStart,
            List<IDataItem> fxwList,
            uint distanceMemElementCount)
            : base(deviceChooser, silentStart)
        {
            if (fxwList == null)
            {
                throw new ArgumentNullException("fxwList");
            }

            _fxwList = fxwList;
            _distanceMemElementCount = distanceMemElementCount;

            this.GenerateMems();
            this.FillMems();
            this.WriteMems();
        }

        public void AllocateAccumulator(
            ulong accumulatorItemCount)
        {
            if (this.AccumMem != null)
            {
                throw new InvalidOperationException("Accumulator already allocated.");
            }

            this.AccumulatorActualItemCount = accumulatorItemCount;

            this.AccumMem = this.CreateByteMem(
                accumulatorItemCount * (sizeof(int) * 2 + sizeof(float)),
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            this.AccumMem.Array.Fill((byte)255);
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
            IndexMem.Array[0] = 0;
        }

        private void GenerateMems()
        {
            FxwMem = this.CreateHalfMem(
                _fxwList.Count * _fxwList[0].Input.Length,
                MemFlags.CopyHostPtr | MemFlags.ReadOnly);

            IndexMem = this.CreateUintMem(
                1,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            DistanceMem = this.CreateByteMem(
                _distanceMemElementCount * (sizeof(int) *  2 + sizeof(float)),
                MemFlags.CopyHostPtr | MemFlags.WriteOnly);

            AccumMem = null;
        }
    }
}