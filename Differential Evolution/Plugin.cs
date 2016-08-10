using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using HeuristicLab.PluginInfrastructure;

namespace HeuristicLab.Algorithms.DifferentialEvolution
{
    [Plugin("HeuristicLab.Algorithms.DifferentialEvolution", "Provides an implementation of DE algorithm", "3.3.9.0")]
    [PluginFile("HeuristicLab.Algorithms.DifferentialEvolution.dll", PluginFileType.Assembly)]
    public class Plugin : PluginBase
    {
    }
}
