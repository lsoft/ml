using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MyNN.Common.Data.Visualizer.Factory
{
    public interface IVisualizerFactory
    {
        IVisualizer CreateVisualizer(
            int dataCount
            );
    }
}
