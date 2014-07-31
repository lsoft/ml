/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 */
inline void WarpReductionToFirstElement(
    __local float *partialDotProduct)
{
#define WARP_SIZE 32

      // Thread local ID within a warp
      uint id = get_local_id(0) & (WARP_SIZE - 1); 

      // Each warp reduces 64 (default) consecutive elements
      float warpResult = 0.0f;
      if (get_local_id(0) < get_local_size(0)/2 )
      {
          volatile __local float* p = partialDotProduct + 2 * get_local_id(0) - id;
          p[0] += p[32];
          p[0] += p[16];
          p[0] += p[8];
          p[0] += p[4];
          p[0] += p[2];
          p[0] += p[1];
          warpResult = p[0];
      }

      // Synchronize to make sure each warp is done reading
      // partialDotProduct before it is overwritten in the next step
      barrier(CLK_LOCAL_MEM_FENCE);

      // The first thread of each warp stores the result of the reduction
      // at the beginning of partialDotProduct
      if (id == 0)
      {
         partialDotProduct[get_local_id(0) / WARP_SIZE] = warpResult;
      }

      // Synchronize to make sure each warp is done writing to
      // partialDotProduct before it is read in the next step
      barrier(CLK_LOCAL_MEM_FENCE);

      // Number of remaining elements after the first reduction
      uint size = get_local_size(0) / (2 * WARP_SIZE);

      // get_local_size(0) is less or equal to 512 on NVIDIA GPUs, so
      // only a single warp is needed for the following last reduction
      // step
      if (get_local_id(0) < size / 2)
      {
         volatile __local float* p = partialDotProduct + get_local_id(0);

         if (size >= 8)
            p[0] += p[4];
         if (size >= 4)
            p[0] += p[2];
         if (size >= 2)
            p[0] += p[1];
      }

}