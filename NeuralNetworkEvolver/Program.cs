using System;

namespace NeuralNetworkEvolver
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create dataset representing XOR
            Dataset xorDataset = new Dataset();
            xorDataset.Add(new double[] { 0, 0 }, new double[] { 0 });
            xorDataset.Add(new double[] { 0, 1 }, new double[] { 1 });
            xorDataset.Add(new double[] { 1, 0 }, new double[] { 1 });
            xorDataset.Add(new double[] { 1, 1 }, new double[] { 0 });

            // Evolve network to recognise XOR
            NeuralNetwork xorNetwork = new NeuralNetwork(2, 2, 1);
            EvolveNetwork en = new EvolveNetwork(xorNetwork, xorDataset);
            en.RunEvolution();
            Console.ReadLine();

            // Create dataset representing XOR with 3 inputs
            Dataset xor3Dataset = new Dataset();
            xor3Dataset.Add(new double[] { 0, 0, 0 }, new double[] { 0 });
            xor3Dataset.Add(new double[] { 0, 0, 1 }, new double[] { 1 });
            xor3Dataset.Add(new double[] { 0, 1, 0 }, new double[] { 1 });
            xor3Dataset.Add(new double[] { 0, 1, 1 }, new double[] { 0 });
            xor3Dataset.Add(new double[] { 1, 0, 0 }, new double[] { 1 });
            xor3Dataset.Add(new double[] { 1, 0, 1 }, new double[] { 0 });
            xor3Dataset.Add(new double[] { 1, 1, 0 }, new double[] { 0 });
            xor3Dataset.Add(new double[] { 1, 1, 1 }, new double[] { 0 });

            // Evolve network to recognise XOR with 3 inputs
            NeuralNetwork xor3Network = new NeuralNetwork(3, 3, 1);
            EvolveNetwork en3 = new EvolveNetwork(xor3Network, xor3Dataset);
            en3.Generations = 10000;
            en3.PopulationSize = 1000;
            en3.RunEvolution();
            Console.ReadLine();
        }
    }    
}
