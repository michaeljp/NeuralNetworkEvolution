using System;
using System.Collections.Generic;

namespace NeuralNetworkEvolver
{
    class Dataset
    {
        public List<double[]> Inputs;
        public List<double[]> ExpectedOutputs;

        public Dataset()
        {
            this.Inputs = new List<double[]>();
            this.ExpectedOutputs = new List<double[]>();
        }
        public void Add(double[] input, double[] expectedOutput)
        {
            this.Inputs.Add(input);
            this.ExpectedOutputs.Add(expectedOutput);
        }

        public int NumberOfOutputs()
        {
            if (this.ExpectedOutputs != null)
                return this.ExpectedOutputs[0].Length;
            else
                return 0;
        }
    }
}
