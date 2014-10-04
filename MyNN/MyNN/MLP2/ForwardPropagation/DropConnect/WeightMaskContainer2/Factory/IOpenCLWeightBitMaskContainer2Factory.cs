using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MyNN.MLP2.Structure.Layer;
using OpenCL.Net.Wrapper;

namespace MyNN.MLP2.ForwardPropagation.DropConnect.WeightMaskContainer2.Factory
{
    public interface IOpenCLWeightBitMaskContainer2Factory
    {
        IOpenCLWeightBitMaskContainer2 CreateContainer2(
            CLProvider clProvider,
            ILayerConfiguration previousLayerConfiguration,
            ILayerConfiguration currentLayerConfiguration
            );
    }
}
