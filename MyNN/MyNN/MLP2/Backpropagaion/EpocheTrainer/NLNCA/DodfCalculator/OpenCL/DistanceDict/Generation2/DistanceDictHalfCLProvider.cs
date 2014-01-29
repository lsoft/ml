using System;
using System.Collections.Generic;
using MyNN.Data;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.DeviceChooser;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Generation2
{
    public class DistanceDictHalfCLProvider : CLProvider
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

        public DistanceDictHalfCLProvider(
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
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadOnly);

            IndexMem = this.CreateIntMem(
                1,
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);

            DistanceMem = this.CreateFloatMem(
                _distanceMemElementCount * 3,
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.WriteOnly);
        }
    }
}