using System;

namespace NeuralNetworkEvolver
{
    class Neuron
    {
        public double[] Weights { get; set; }
        public Neuron(int numberOfWeights)
        {
            this.Weights = new double[numberOfWeights];
        }
        public double SumInputs(double[] inputs)
        {
            double sumOfInputs = 0.0;
            sumOfInputs += 1 * this.Weights[0];  // Add bias
            for (int i = 0; i < inputs.Length; i++)
            {
                sumOfInputs += inputs[i] * this.Weights[i + 1];
            }
            return sumOfInputs;
        }
    }

    class Layer
    {
        public Neuron[] Neurons { get; set; }
        public double[] Inputs { get; set; }
        public Layer(int numberOfNeurons, int numberOfInputs)
        {
            this.Neurons = new Neuron[numberOfNeurons];
            this.Inputs = new double[numberOfInputs];
            int numberOfWeightsPerNeuron = numberOfInputs + 1;  // +1 for bias
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neuron newNeuron = new Neuron(numberOfWeightsPerNeuron);
                this.Neurons[i] = newNeuron;
            }
        }

        public double[] Activate()
        {
            double[] output = new double[this.Neurons.Length];
            for (int neuronIndex = 0; neuronIndex < this.Neurons.Length; neuronIndex++)
            {
                var inputSum = this.Neurons[neuronIndex].SumInputs(this.Inputs);
                output[neuronIndex] = this.ActivationFunction(inputSum);
            }
            return output;
        }
        public double ActivationFunction(double inputSum)
        {
            // Steepened Sigmoid Function
            return 1.0 / (1.0 + Math.Exp(-3 * inputSum));
        }
    }

    class NeuralNetwork
    {
        public Layer[] Layers { get; set; }
        public int NumberOfWeights { get; set; }

        public NeuralNetwork(params int[] layers)
        {
            this.Layers = new Layer[layers.Length - 1];
            int numberOfWeights = 0;
            for (int i = 1; i < layers.Length; i++)
            {
                Layer newLayer = new Layer(layers[i], layers[i - 1]);
                this.Layers[i - 1] = newLayer;

                numberOfWeights += layers[i] * (layers[i - 1] + 1);  // +1 for bias
            }
            this.NumberOfWeights = numberOfWeights;
        }

        public void SetWeights(double[] genome)
        {
            int genomeIndex = 0;
            for (int layerIndex = 0; layerIndex < this.Layers.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < this.Layers[layerIndex].Neurons.Length; neuronIndex++)
                {
                    for (int weightIndex = 0; weightIndex < this.Layers[layerIndex].Neurons[neuronIndex].Weights.Length; weightIndex++)
                    {
                        this.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] = genome[genomeIndex];
                        genomeIndex++;
                    }
                }
            }
        }

        public void SetInput(double[] input)
        {
            this.Layers[0].Inputs = input;
        }

        public double[] Activate()
        {
            for (int layerIndex = 1; layerIndex < this.Layers.Length; layerIndex++)
            {
                this.Layers[layerIndex].Inputs = this.Layers[layerIndex - 1].Activate();
            }
            return this.Layers[this.Layers.Length - 1].Activate();
        }
    }
}
