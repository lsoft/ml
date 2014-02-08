using System;
using System.Collections.Generic;
using MyNN.Data;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.DeviceChooser;
using OpenCL.Net.Wrapper.Mem;

namespace MyNN.MLP2.Backpropagation.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation4
{
    /// <summary>
    /// Correct OpenCL provider for generation #4 distance providers that consumes HALF input representations.
    /// </summary>
    public class OpenCLDistanceDictHalfProvider : CLProvider
    {
        private readonly List<DataItem> _fxwList;
        private readonly int _distanceMemElementCount;

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

        public MemFloat AccumMem
        {
            get;
            private set;
        }

        public OpenCLDistanceDictHalfProvider(
            IDeviceChooser deviceChooser,
            bool silentStart,
            List<DataItem> fxwList,
            int distanceMemElementCount)
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
            long accumulatorItemCount)
        {
            if (this.AccumMem != null)
            {
                throw new InvalidOperationException("Accumulator already allocated.");
            }

            this.AccumMem = this.CreateFloatMem(
                accumulatorItemCount * 3,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);
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

            IndexMem = this.CreateIntMem(
                1,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            DistanceMem = this.CreateFloatMem(
                _distanceMemElementCount * 3,
                MemFlags.CopyHostPtr | MemFlags.WriteOnly);

            AccumMem = null;
        }
    }
}