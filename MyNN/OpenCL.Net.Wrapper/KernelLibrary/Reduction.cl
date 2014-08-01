inline void WarpReductionToFirstElement(
    __local float *scratch)
{
/*
	//тупейшая реализация, но работает всегда
	if(get_local_id(0) == 0)
	{
		float sum = 0;
		for(int c = 0; c < get_local_size(0); c++)
		{
			sum += scratch[c];
		}
		scratch[0] = sum;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
//*/

    // Perform parallel reduction
    int local_index = get_local_id(0);
    int local_size = get_local_size(0) + (get_local_size(0) % 2);
	int current_local_size = get_local_size(0);

    for(int offset = local_size / 2; offset > 0; offset = (offset + (offset > 1 ? 1 : 0)) / 2)
    {
        if (local_index < offset)
        {
			int other_index = local_index + offset;

			float other = 0;
			if(other_index < current_local_size)
			{
	            other = scratch[other_index];
			}

            scratch[local_index] += other;

        }

        barrier(CLK_LOCAL_MEM_FENCE);

		current_local_size = (current_local_size + 1) / 2;
    }
//*/
}
