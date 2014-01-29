﻿using System;
using System.Collections.Generic;
using MyNN.Data;
using OpenCL.Net.OpenCL;
using OpenCL.Net.OpenCL.DeviceChooser;
using OpenCL.Net.OpenCL.Mem;
using OpenCL.Net.Platform;

namespace MyNN.MLP2.Backpropagaion.EpocheTrainer.NLNCA.DodfCalculator.OpenCL.DistanceDict.Half
{
    public class DistanceDictHalfCLProvider : CLProvider
    {
        private readonly List<DataItem> _fxwList;

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

        public DistanceDictHalfCLProvider(List<DataItem> fxwList)
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
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadOnly);

            IndexMem = this.CreateIntMem(
                _fxwList.Count,
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadOnly);

            var totalCount = (_fxwList.Count + 1) * _fxwList.Count / 2;

            DistanceMem = this.CreateFloatMem(
                totalCount,
                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.WriteOnly);
        }
    }
}