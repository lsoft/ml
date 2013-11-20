using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MyNN.OpenCL;
using MyNN.OpenCL.Mem;
using OpenCL.Net;

namespace MyNN.NeuralNet.Train.Algo.NLNCA.DodfCalculator
{
//недописано
//    public class DodfCalculatorOpenCL
//    {
//        private readonly  CLProvider _clProvider;

//        public DodfCalculatorOpenCL()
//        {
//            _clProvider = new CLProvider(true);
//        }


//    }

//    public class PabCalculatorOpenCL
//    {
//        private readonly CLProvider _clProvider;

//        private readonly Kernel _createDistanceKernel;

//        private readonly MemFloat _distanceMem;
//        private readonly MemFloat _fxwMem;

//        private readonly int _itemLength;
//        private readonly int _itemCount;


//        public PabCalculatorOpenCL(
//            List<DataItem> fxwList)
//        {
//            _clProvider = new CLProvider(true);
//            _itemLength = fxwList[0].Input.Length;
//            _itemCount = fxwList.Count;

//            var kernelSource = _kernelSource.Replace("{0}", fxwList[0].Input.Length.ToString());

//            _createDistanceKernel = _clProvider.CreateKernel(kernelSource, "CreateDistanceMatrix");

//            _distanceMem = _clProvider.CreateFloatMem(
//                _itemCount * _itemCount,
//                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadOnly);
//            _distanceMem.Write(BlockModeEnum.Blocking);

//            _fxwMem = _clProvider.CreateFloatMem(
//                _itemCount * _itemLength,
//                Cl.MemFlags.CopyHostPtr | Cl.MemFlags.ReadWrite);
//            var index = 0;
//            foreach (var d in fxwList)
//            {
//                Array.Copy(
//                    d.Input,
//                    0,
//                    _fxwMem.Array,
//                    index,
//                    _itemLength);

//                index += _itemLength;
//            }
//            _fxwMem.Write(BlockModeEnum.Blocking);

//            _createDistanceKernel
//                .SetKernelArgMem(0, _fxwMem)
//                .SetKernelArgMem(1, _distanceMem)
//                .EnqueueNDRangeKernel(_itemCount);

//            _clProvider.QueueFinish();

//            _distanceMem.Read(BlockModeEnum.Blocking);

//        }

//        private const string _kernelSource = @"
//typedef struct
//{
//    float Input[{0}];
//} FxItem;
//
//float GetDistanceab(
//    __global read_only FxItem* fxwList,
//    int a,
//    int b)
//{
//    //__const 
//        int ValuesLength = {0};
//
//    float result = 0;
//    for (int cc = 0; cc < ValuesLength; cc++)
//    {
//        float diff = fxwList[a].Input[cc] - fxwList[b].Input[cc];
//        result += diff * diff;
//    }
//
//    return result;
//}
//
//void GetDab(
//    __global FxItem* read_only fxwList,
//    __global float * write_only result,
//    int a,
//    int b)
//{
//    //__const 
//        int ValuesLength = {0};
//
//    for (int cc = 0; cc < ValuesLength; cc++)
//    {
//        result[cc] = fxwList[a].Input[cc] - fxwList[b].Input[cc];
//    }
//}
//
//__kernel void CreateDistanceMatrix(
//    __global read_only FxItem* fxwList,
//    __global write_only float* result)
//{
//    //__const 
//        int ValuesLength = {0};
//    
//    int a = get_global_id(0);
//    int fxLength = get_global_size(0);
//
//    int startIndex = a * fxLength;
//
//    result[startIndex + a] = 0;
//    for (int dd = a + 1; dd < fxLength; dd++)
//    {
//        float distance = GetDistanceab(fxwList, a, dd);
//        result[startIndex + dd] = distance;
//    }
//
//}
//
//
//";
//    }
}
