using System;

namespace NeuralNetworkEvolver
{
    class Individual
    {
        public double Fitness { get; set; }
        public double[] Genome { get; set; }
    }

    class EvolveNetwork
    {
        private NeuralNetwork network { get; set; }
        private Individual[] population { get; set; }
        public int PopulationSize { get; set; }
        public int TournamentSelectionNumber { get; set; }
        public double CrossoverRate { get; set; }
        public double MutationRate { get; set; }
        private int genomeLength { get; set; }
        private Random rand { get; set; }
        public int Generations { get; set; }
        private double bestIndividualFitness { get; set; }
        public Dataset Data { get; set; }

        public EvolveNetwork(NeuralNetwork network, Dataset dataset)
        {
            this.network = network;
            this.genomeLength = this.network.NumberOfWeights;
            this.PopulationSize = 50;
            this.population = new Individual[this.PopulationSize];
            this.rand = new Random(10);
            this.TournamentSelectionNumber = 5;
            this.CrossoverRate = 0.9;
            this.MutationRate = 0.9;
            this.Generations = 20000;
            this.Data = dataset;
            this.bestIndividualFitness = 0.0;

            createPopulation(this.PopulationSize);
        }

        /* Create a population of Individuals with random genomes */
        private void createPopulation(int populationSize)
        {
            for (int i = 0; i < populationSize; i++)
            {
                var newIndividual = new Individual();
                newIndividual.Genome = randomGenome();
                newIndividual.Fitness = evaluateFitness(newIndividual.Genome);
                this.population[i] = newIndividual;
            }
        }

        /* Create a new genome with random genes */
        private double[] randomGenome()
        {
            double[] genome = new double[this.genomeLength];
            for (int i = 0; i < genome.Length; i++)
                genome[i] = uniformRandomNumber();
            return genome;
        }

        /*
         * Evaluate the fitness of a particular genome. Apply the genome to the weights of the 
         * network, and determine the mean squared error for the dataset.
         */
        private double evaluateFitness(double[] testGenome)
        {
            this.network.SetWeights(testGenome);

            // Calculate Mean Squared Error
            double squaredError = 0.0;
            int numberOfOutputs = this.Data.NumberOfOutputs();
            double[] networkOutputs = new double[numberOfOutputs];
            double error = 0;
            for (int row = 0; row < this.Data.Inputs.Count; row++)
            {
                network.SetInput(Data.Inputs[row]);
                networkOutputs = network.Activate();

                for (int col = 0; col < numberOfOutputs; col++)
                {
                    error = networkOutputs[col] - this.Data.ExpectedOutputs[row][col];
                    squaredError += error * error;
                }
            }
            return -squaredError / this.Data.Inputs.Count;
        }

        public void PrintResults()
        {
            // Get best genome
            int bestIndividualIndex = 0;
            for (int i = 0; i < this.population.Length; i++)
                if (this.population[i].Fitness > this.population[bestIndividualIndex].Fitness)
                    bestIndividualIndex = i;
            this.network.SetWeights(this.population[bestIndividualIndex].Genome);

            int numberOfOutputs = this.Data.NumberOfOutputs();
            double[] networkOutputs = new double[numberOfOutputs];

            for (int row = 0; row < this.Data.Inputs.Count; row++)
            {
                network.SetInput(Data.Inputs[row]);
                networkOutputs = network.Activate();

                Console.Write("\nOutput:");
                for (int col = 0; col < numberOfOutputs; col++)
                    Console.Write(" {0:F2}", networkOutputs[col]);
                Console.Write("\tExpected:");
                for (int col = 0; col < numberOfOutputs; col++)
                    Console.Write(" {0}", this.Data.ExpectedOutputs[row][col]);
            }
        }

        /* Return a uniform random number between -1 and 1 */
        private double uniformRandomNumber()
        {
            double nextRandom = this.rand.Next(Int32.MinValue, Int32.MaxValue);
            return nextRandom / Int32.MaxValue;
        }

        /* Return a gaussian random number with mean 0 and standard deviation 1 */
        private double gaussianRandomNumber()
        {
            return Math.Sqrt(
                -2.0 * Math.Log(this.rand.NextDouble())) *
                Math.Cos(2.0 * Math.PI * this.rand.NextDouble()
            );
        }

        /*
         * Select a parent genome for evolution through tournament selection. A subset of the population
         * is chosen at random and the fittest genome from the subset is returned.
         */
        public double[] tournamentSelection()
        {
            int bestCandidateIndex = rand.Next() % this.population.Length;
            for (int i = 0; i < this.TournamentSelectionNumber; i++)
            {
                int newCandidateIndex = rand.Next() % this.population.Length;
                if (this.population[newCandidateIndex].Fitness > this.population[bestCandidateIndex].Fitness)
                    bestCandidateIndex = newCandidateIndex;
            }
            return this.population[bestCandidateIndex].Genome;
        }

        /* Mutate a child genome, implementing small changes to the genes. */
        public double[] mutate(double[] childGenome)
        {
            int numberOfGenesToMutate = 3;
            for (int i = 0; i < numberOfGenesToMutate; i++)
            {
                int randIndex = Convert.ToInt16(Math.Abs(uniformRandomNumber()) * (this.genomeLength - 1));
                childGenome[randIndex] += gaussianRandomNumber();
            }
            return childGenome;
        }

        /* Produce genome for a child as a two-point crossover of the genome of two parents. */
        public double[] crossover(double[] parentOneGenome, double[] parentTwoGenome)
        {
            // Two-point crossover
            double[] childGenome = new double[this.genomeLength];
            int crossoverPoint1 = (int)(Math.Abs(uniformRandomNumber()) * (this.genomeLength));
            int crossoverPoint2 = (int)(
                    (this.genomeLength - crossoverPoint1) *
                    Math.Abs(uniformRandomNumber()) + crossoverPoint1
                );
            for (int i = 0; i < this.genomeLength; i++)
            {
                if (i < crossoverPoint1 || i > crossoverPoint2)
                    childGenome[i] = parentOneGenome[i];
                else
                    childGenome[i] = parentTwoGenome[i];
            }
            return childGenome;
        }

        /* Run evolution steps for the specified number of generations. */
        public void RunEvolution()
        {
            for (int generation = 0; generation < this.Generations; generation++)
            {
                if (generation % 100 == 0)
                    Console.WriteLine("Generation: {0} Best Fitness: {1}",
                        generation, this.bestIndividualFitness);
                runEvolutionStep();
            }
            this.PrintResults();
        }

        /*
         * Run evolution for one generation, creating a new child for the population. This child
         * may be a copy of an existing Individual, a crossover of two Individuals, or a mutation
         * of either. This child will be added to the population if it has sufficient fitness.
         */
        public void runEvolutionStep()
        {
            double[] childGenome = new double[this.genomeLength];

            // select parent 1
            double[] parentOneGenome = tournamentSelection();

            //crossover
            if (rand.NextDouble() < this.CrossoverRate)
            {
                double[] parentTwoGenome = tournamentSelection();
                childGenome = crossover(parentOneGenome, parentTwoGenome);
            }

            //mutate child
            if (rand.NextDouble() < this.MutationRate)
                childGenome = mutate(childGenome);

            //evaluate child fitness
            double childFitness = evaluateFitness(childGenome);

            updatePopulation(childGenome, childFitness);
        }

        /*
         * Update the population by swapping the Individual with lowest fitness with the 
         * newly created child if the child fitness is equal or higher.
         */
        public void updatePopulation(double[] childGenome, double childFitness)
        {
            // Find Individual with lowest fitness (and Individual with best fitness)
            int bestIndividualIndex = 0;
            int worstIndividualIndex = 0;
            for (int i = 0; i < this.population.Length; i++)
            {
                if (this.population[i].Fitness > this.population[bestIndividualIndex].Fitness)
                    bestIndividualIndex = i;
                if (this.population[i].Fitness < this.population[worstIndividualIndex].Fitness)
                    worstIndividualIndex = i;
            }
            this.bestIndividualFitness = this.population[bestIndividualIndex].Fitness;

            // Swap worst Individual for new child
            if (this.population[worstIndividualIndex].Fitness <= childFitness)
            {
                this.population[worstIndividualIndex].Genome = childGenome;
                this.population[worstIndividualIndex].Fitness = childFitness;
            }
        }
    }
}
