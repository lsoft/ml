using System;
using OpenCL.Net;
using OpenCL.Net.Wrapper;
using OpenCL.Net.Wrapper.Mem;
using OpenCL.Net.Wrapper.Mem.Data;
using Kernel = OpenCL.Net.Wrapper.Kernel;

namespace MyNN.MLP.Transposer
{
    public class TransposerNvidia : IOpenCLTransposer
    {
        private const int BLOCK_DIM = 16;

        private readonly CLProvider _clProvider;
        private readonly MemFloat _source;
        private readonly int _width;
        private readonly int _height;
        
        private MemFloat _destination;
        
        private Kernel _transposeKernel;

        public MemFloat Destination
        {
            get
            {
                return _destination;
            }
        }

        public TransposerNvidia(
            CLProvider clProvider,
            MemFloat source,
            int width,
            int height)
        {
            if (clProvider == null)
            {
                throw new ArgumentNullException("clProvider");
            }
            if (source == null)
            {
                throw new ArgumentNullException("source");
            }

            _clProvider = clProvider;
            _source = source;
            _width = width;
            _height = height;

            PrepareInfrastructure();
        }

        #region prepare infrastructure

        private void PrepareInfrastructure()
        {
            _destination = _clProvider.CreateFloatMem(
                _width * _height,
                MemFlags.CopyHostPtr | MemFlags.ReadWrite);

            _destination.Write(BlockModeEnum.Blocking);

            var kernelSource = KernelSource.Replace("{BLOCK_DIM}", BLOCK_DIM.ToString());
            _transposeKernel = _clProvider.CreateKernel(
                kernelSource,
                "NVTranspose");
        }


        #endregion

        public void Transpose()
        {
            var glob = new int[]
            {
                ShrRoundUp(BLOCK_DIM, _width),
                ShrRoundUp(BLOCK_DIM, _height)
            };

            var loc = new int[]
            {
                BLOCK_DIM,
                BLOCK_DIM,
            };

            _transposeKernel
                .SetKernelArgMem(0, _destination)
                .SetKernelArgMem(1, _source)
                .SetKernelArg(2, 4, 0)
                .SetKernelArg(3, 4, _width)
                .SetKernelArg(4, 4, _height)
                .SetKernelArgLocalMem(5, (BLOCK_DIM + 1) * BLOCK_DIM * 4)
                .EnqueueNDRangeKernel(
                    glob,
                    loc);
        }

        private int ShrRoundUp(int groupSize, int globalSize)
        {
            int r = globalSize % groupSize;

            if (r == 0)
            {
                return globalSize;
            }
            else
            {
                return globalSize + groupSize - r;
            }
        }


        private const string KernelSource = @"
/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define BLOCK_DIM {BLOCK_DIM}

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.
__kernel void NVTranspose(
    __global float *odata,
    __global float *idata,
    int offset,
    int width,
    int height,
    __local float* block)
{
    // read the matrix tile into shared memory
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);

    if((xIndex + offset < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex + offset;
        block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // write the transposed matrix tile to global memory
    xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
    yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
    if((xIndex < height) && (yIndex + offset < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];
    }
//*/
}
";
    }
}