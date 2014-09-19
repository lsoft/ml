/*
//naive implementation (for testing only)
//local size must be equal partialDotProduct size
inline void WarpReductionToFirstElement(
    __local float *partialDotProduct)
{
	if(get_local_id(0) == 0)
	{
		float sum = 0;
		for(int c = 0; c < get_local_size(0); c++)
		{
			sum += partialDotProduct[c];
		}
		partialDotProduct[0] = sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}
//*/


//reduction with 2x rate
//local size must be equal partialDotProduct size
inline void WarpReductionToFirstElement(
    __local float *partialDotProduct)
{

    // Perform parallel reduction
    int local_index = get_local_id(0);
    int local_size = ((get_local_size(0) + 1) / 2) * 2;
	int current_local_size = get_local_size(0);

    for(int offset = local_size / 2; offset > 0; offset = (offset + (offset > 1 ? 1 : 0)) / 2)
    {
        if (local_index < offset)
        {
			int other_index = local_index + offset;
			if(other_index < current_local_size)
			{
		        partialDotProduct[local_index] += partialDotProduct[other_index];
			}
        }

        barrier(CLK_LOCAL_MEM_FENCE);

		current_local_size = (current_local_size + 1) / 2;
    }
}
//*/


/*
//reduction with 4x rate (no palpable benefit on Intel CPU, Intel HD Graphics 4400, NVidia GT 730m)
//local size must be equal partialDotProduct size
inline void WarpReductionToFirstElement(
    __local float *partialDotProduct)
{
    // Perform parallel reduction
    int local_index = get_local_id(0);
    int local_size = ((get_local_size(0) + 3) / 4) * 4;
	int current_local_size = get_local_size(0);

    for(int offset = local_size / 4; offset > 0; offset = (offset + (offset > 1 ? 3 : 0)) / 4)
    {
        if (local_index < offset)
        {
			float accum = partialDotProduct[local_index];

			int other_index0 = local_index + offset;
			if(other_index0 < current_local_size)
			{
		        accum  += partialDotProduct[other_index0];

				int other_index1 = other_index0 + offset;
				if(other_index1 < current_local_size)
				{
					accum += partialDotProduct[other_index1];

					int other_index2 = other_index1 + offset;
					if(other_index2 < current_local_size)
					{
						accum  += partialDotProduct[other_index2];

						int other_index3 = other_index2 + offset;
						if(other_index3 < current_local_size)
						{
							accum += partialDotProduct[other_index3];
						}
					}
				}

				partialDotProduct[local_index] = accum;
			}
		}

        barrier(CLK_LOCAL_MEM_FENCE);

		current_local_size = (current_local_size + 3) / 4;
    }
}
//*/

