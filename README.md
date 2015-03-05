NeuralNetworkEvolution
======================

Implementation of an evolutionary algorithm for training a multi-layer neural network.

Usage
-----

1. Create a neural network.

   ```cs
   // Create a network with 2 inputs, 2 hidden layers with 20 neurons each, and 1 output
   NeuralNetwork network = new NeuralNetwork(2, 20, 20, 1);
   ```
2. Create a dataset which will be used for for guiding evolution.

   ```cs
   // Create dataset representing XOR
   Dataset dataset = new Dataset();
   dataset.Add(new double[] { 0, 0 }, new double[] { 0 });
   dataset.Add(new double[] { 0, 1 }, new double[] { 1 });
   dataset.Add(new double[] { 1, 0 }, new double[] { 1 });
   dataset.Add(new double[] { 1, 1 }, new double[] { 0 });
   ````
3. Create a network evolver, supplying the network to be evolved and the dataset.
   
   ```cs
   EvolveNetwork en = new EvolveNetwork(network, dataset);
   ```
4. The network evolver has default parameters, but these can also be adjusted.

   ```cs
   en.Generations = 2000;
   en.PopulationSize = 1000;
   en.MutationRate = 0.9;
   en.CrossoverRate = 0.9;
   en.TournamentSelectionNumber = 5;
   en.NumberOfGenesToMutate = 3;
   ```
5. Run the evolution.

   ```cs
   en.RunEvolution();
   ```
   Results

   ```
   Generation: 100 Best Fitness: -0.248749
   ...
   ...
   Generation: 1800 Best Fitness: -0.000000
   Generation: 1900 Best Fitness: -0.000000
   
   Output: 0.00   Expected: 0
   Output: 1.00   Expected: 1
   Output: 1.00   Expected: 1
   Output: 0.00   Expected: 0
   ```

   
