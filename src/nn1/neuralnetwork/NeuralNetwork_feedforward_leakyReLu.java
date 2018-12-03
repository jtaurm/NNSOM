/*
 * The MIT License
 *
 * Copyright 2018 Jacob Cornelius Mosebo.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package nn1.neuralnetwork;

import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 */
public class NeuralNetwork_feedforward_leakyReLu 
{
    int Input_height;
    int Matrix_height;
    int Matrix_depth;
    int Output_height;
  
    double[] Neuron_input_bias;
    double[] Neuron_input_activity;
    double[] Neuron_input_descent;
    
    double[][] Axon_input_weights;
    double[][] Axon_input_activity;
    double[][] Axon_input_descent;
    
    double[][] Neuron_matrix_bias;
    double[][] Neuron_matrix_activity;
    double[][] Neuron_matrix_descent;
    
    double[][][] Axon_matrix_weights;
    double[][][] Axon_matrix_activity;
    double[][][] Axon_matrix_descent;
    
    double[] Neuron_output_bias;
    double[] Neuron_output_activity;
    double[] Neuron_output_descent;
    
    double[][] Axon_output_weights;
    double[][] Axon_output_activity;
    double[][] Axon_output_descent;
    
    public void Initialize( int inputs, int outputs, int height, int depth)
    {
        Input_height = inputs;
        Matrix_height = height;
        Matrix_depth = depth;
        Output_height = outputs;
        
        // Setup inputs
        Neuron_input_bias = InitializeRandom(inputs);
        Neuron_input_activity = InitializeRandom(inputs);
        Neuron_input_descent = InitializeRandom(inputs);
        
        Axon_input_weights = InitializeRandom(inputs, height);
        Axon_input_activity = InitializeRandom(inputs, height);
        Axon_input_descent = InitializeRandom(inputs, height);
        
        // Setup matrix
        Neuron_matrix_bias = InitializeRandom(depth, height);
        Neuron_matrix_activity = InitializeRandom(depth, height);
        Neuron_matrix_descent = InitializeRandom(depth, height);
        
        Axon_matrix_weights = InitializeRandom(depth-1, height, height);
        Axon_matrix_activity = InitializeRandom(depth-1, height, height);
        Axon_matrix_descent = InitializeRandom(depth-1, height, height);
        
        // Setup output
        Neuron_output_bias = InitializeRandom(outputs);
        Neuron_output_activity = InitializeRandom(outputs);
        Neuron_output_descent = InitializeRandom(outputs);
        
        Axon_output_weights = InitializeRandom(outputs, height);
        Axon_output_activity = InitializeRandom(outputs, height);
        Axon_output_descent = InitializeRandom(outputs, height);
        
    }
    
    private double[] InitializeRandom( int size )
    {
        double[] rand_arr = new double[size];
        for(int i = 0; i < size; i++)
        {
            double r = Math.random();
            rand_arr[i] = (r > 0.5 ? r - 0.25 : -0.25 - r);
        }
        return rand_arr;
    }
    
    private double[][] InitializeRandom( int size0, int size1 )
    {
        double[][] rand_arr = new double[size0][size1];
        for(int i = 0; i < size0; i++)
        {
            rand_arr[i] = InitializeRandom(size1);
        }
        return rand_arr;
    }
    
    private double[][][] InitializeRandom( int size0, int size1, int size2 )
    {
        double[][][] rand_arr = new double[size0][size1][size2];
        for(int i = 0; i < size0; i++)
        {
            rand_arr[i] = InitializeRandom(size1, size2);
        }
        return rand_arr;
    }
    
    public double[] FeedForward(double[] input_stimulus)
    {
        // Stimulate inputs neurons
        for(int i = 0; i < Input_height; i++)
        {
            Neuron_input_activity[i] = Neuron_input_bias[i]  + input_stimulus[i];
        }
        
        // Stimulate input axons
        for(int i = 0; i < Input_height; i++)
            for(int h = 0; h < Matrix_height; h++)
                Axon_input_activity[i][h] = Neuron_input_activity[i] * Axon_input_weights[i][h];
        
        // Stimulate first layer of the matrix neurons
        System.arraycopy(Neuron_matrix_bias[0], 0, Neuron_matrix_activity[0], 0, Matrix_height);
        
        for(int h = 0; h < Matrix_height; h++)
        {
            // Stimulate with input axon activity
            for(int i = 0; i < Input_height; i++)
                Neuron_matrix_activity[0][h] += Axon_input_activity[i][h];
            // Simulate Leaky ReLu
            for(int i = 0; i < Input_height; i++)
                if(Neuron_matrix_activity[0][h] < 0d)
                    Neuron_matrix_activity[0][h] *= 0.1;
        }
        
        // Stimulate hidden layers
        // Reset activity to bias
        for(int d = 1; d < Matrix_depth; d++)
            System.arraycopy(Neuron_matrix_bias[d], 0, Neuron_matrix_activity[d], 0, Matrix_height);
        
        for(int d = 1; d < Matrix_depth; d++)
        {
            for(int h_n = 0; h_n < Matrix_height; h_n++)
            {
                // Stimulate axons from prev layer
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_activity[d-1][h_n][h_a] = Neuron_matrix_activity[d-1][h_n] * Axon_matrix_weights[d-1][h_n][h_a];
                // Stimulate neurons from axons
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Neuron_matrix_activity[d][h_n] += Axon_matrix_activity[d-1][h_n][h_a];
            }
            // Simulate Leaky ReLu
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                if(Neuron_matrix_activity[d][h_n] < 0d)
                    Neuron_matrix_activity[d][h_n] *= 0.1;
        }
            
        
        // Stimulate output layer axons
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_output_activity[h_o][h_n] = Neuron_matrix_activity[Matrix_depth - 1][h_n] * Axon_output_weights[h_o][h_n];
        
        // Stimulate output layer neurons
        // Reset activity to bias
        System.arraycopy(Neuron_output_bias, 0, Neuron_output_activity, 0, Output_height);
        
        // Stimulate activity
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Neuron_output_activity[h_o] += Axon_output_activity[h_o][h_n];
        // No Leaky ReLu here, output neurons are linear
        
        // Return copy of output neuron activity
        return Arrays.copyOf(Neuron_output_activity, Output_height);
    }
    
    public void BackPropagate(double[] costs)
    {
        // Output layer
        // dAct = cost, neuron is linear: dZ = dAct
        System.arraycopy(costs, 0, Neuron_output_descent, 0, Output_height);
        
        // Output axons
        // dZ = from.activity * to.dZ
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_output_descent[h_o][h_n] = Neuron_matrix_activity[Matrix_depth - 1][h_n] * Neuron_output_descent[h_o];
        
        // Last layer of hidden Neurons
        // Neuron dZ = axon_out[].weight * neuron_out[].dZ
        for(int h_n = 0; h_n < Matrix_height; h_n++)
        {
            Neuron_matrix_descent[Matrix_depth - 1][h_n] = 0d;

            // Calculate Neuron dZ
            for(int h_o = 0; h_o < Output_height; h_o++)
                Neuron_matrix_descent[Matrix_depth - 1][h_n] += Axon_output_weights[h_o][h_n] * Neuron_output_descent[h_o];
            
            // Simulate Leaky ReLu
            for(int h_o = 0; h_o < Output_height; h_o++)
                if(Neuron_output_activity[h_o] < 0d)
                Neuron_matrix_descent[Matrix_depth - 1][h_n] *= 0.1;
        }
        
        // Hidden layers
        for(int d = Matrix_depth - 2; d >= 0; d--)
        {
            for(int h_n = 0; h_n < Matrix_height; h_n++)
            {
                // Calculate Axon dZ = from.activity * to.dZ
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_descent[d][h_n][h_a] = Neuron_matrix_activity[d][h_n] * Neuron_matrix_descent[d + 1][h_n];
                
                // Calculate Neuron dZ = axon_out[].weight * neuron_out[].dZ
                Neuron_matrix_descent[d][h_n] = 0d;
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Neuron_matrix_descent[d][h_n] += Axon_matrix_weights[d][h_n][h_a] * Neuron_matrix_descent[d + 1][h_n];
                
                // Simulate Leaky ReLu
                for(int h_o = 0; h_o < Output_height; h_o++)
                    if(Neuron_matrix_activity[d + 1][h_o] < 0d)
                        Neuron_matrix_descent[d][h_n] *= 0.1;
                
            }
        }
        
        // Input axons
        for(int h_i = 0; h_i < Input_height; h_i++)
            // Calculate Axon dZ = from.activity * to.dZ
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_input_descent[h_i][h_n] = Neuron_input_activity[h_i] * Neuron_matrix_descent[0][h_n];
        
        // Input neurons
        // Neuron dZ = axon_out[].weight * neuron_out[].dZ
        for(int h_i = 0; h_i < Input_height; h_i++)
        {
            Neuron_input_descent[h_i] = 0d;
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Neuron_input_descent[h_i] += Axon_input_weights[h_i][h_n] * Neuron_matrix_descent[0][h_n];
            
            // Simulate Leaky ReLu
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                if(Neuron_matrix_activity[0][h_n] < 0d)
                    Neuron_input_descent[h_i] *= 0.1;
        }
        
    }
    
    public void UpdateWeights(double learning_rate)
    {
        // Update output
        for(int h_o = 0; h_o < Output_height; h_o++)
            Neuron_output_bias[h_o] = Neuron_output_descent[h_o] * learning_rate;
        
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_output_weights[h_o][h_n] = Axon_output_descent[h_o][h_n] * learning_rate;
        
        // Update hidden layers
        for(int d = 0; d < Matrix_depth - 1; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_weights[d][h_n][h_a] = Axon_matrix_descent[d][h_n][h_a] * learning_rate;
        
        for(int d = 0; d < Matrix_depth; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Neuron_matrix_bias[d][h_n] = Neuron_matrix_descent[d][h_n] * learning_rate;

        // Update output
        for(int h_i = 0; h_i < Input_height; h_i++)
            Neuron_input_bias[h_i] = Neuron_input_descent[h_i] * learning_rate;
        
        for(int h_i = 0; h_i < Input_height; h_i++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_input_weights[h_i][h_n] = Axon_input_descent[h_i][h_n] * learning_rate;
    }
    
    public void Train(double[][] training_data, int[] ColumnIOMap, int rep_count, double learning_rate)
    {
        int rowCount = training_data.length;
        
        int input_size = 0;
        int output_size = 0;
        for(int i = 0; i < ColumnIOMap.length; i++)
        {
            if(ColumnIOMap[i] == 1)
                input_size++;
            
            if(ColumnIOMap[i] == -1)
                output_size++;
        }
        
        ArrayList<double[]> Errors = new ArrayList<>();
        
        // for each repeatition of data set
        for(int rep = 0; rep < rep_count; rep++)
        {
            // Generate indices list
            int[] idx_order = new int[rowCount];
            for(int i = 0; i < rowCount; i++)
                idx_order[i] = i;
            
            // Randomize indices list
            for(int i = 0; i < rowCount; i++)
            {
                int p = (int) (Math.random() * rowCount);
                int swap = idx_order[i];
                idx_order[i] = idx_order[p];
                idx_order[p] = swap;
            }
            
            // Run through data set / table
            for(int r = 0; r < rowCount; r++)
            {
                // Build stimulation and response variable arrays
                double[] stimlation = new double[input_size];
                double[] response = new double[output_size];
                double[] errors = new double[output_size];
                double[] predictions;
                int inputIdx = 0;
                int outputIdx = 0;

                for(int p = 0; p < ColumnIOMap.length; p++)
                {
                    if(ColumnIOMap[p] == 1)
                    {
                        stimlation[inputIdx] = training_data[idx_order[r]][p];
                        inputIdx++;
                    }
                    else
                    {
                        response[outputIdx] = training_data[idx_order[r]][p];
                        outputIdx++;
                    }
                }

                // Feed stimulus to learning machine
                predictions = FeedForward(stimlation);

                // Calc cost
                for (outputIdx = 0; outputIdx < output_size; outputIdx++) 
                {
                    errors[outputIdx] = response[outputIdx] - predictions[outputIdx];
                }

                Errors.add(errors);

                // Back propagate costs
                BackPropagate(errors);
                
                // Update weights
                UpdateWeights(learning_rate);

            }
            
            // Calc training MSE
            double[] eSum = new double[output_size];
            Arrays.fill(eSum, 0d);
            
            for(int i = 0; i < Errors.size(); i++)
            {
                for (int outputIdx = 0; outputIdx < output_size; outputIdx++) 
                    eSum[outputIdx] += Errors.get(i)[outputIdx] * Errors.get(i)[outputIdx];
            }
            
            for (int outputIdx = 0; outputIdx < output_size; outputIdx++) 
                eSum[outputIdx] = eSum[outputIdx] / Errors.size();
            
            Errors.clear();
            
            // Show progress
            if (rep % 50 == 0)
                System.out.println( "\tlr: " + learning_rate + "\trep: " + rep + "\ttraining MSE: " + eSum[0] + " " + eSum[1] );
            
        }
        
    }
    
    
    
    
    public double[][] Predict(double[][] training_data, int[] ColumnIOMap)
    {
        
        
        int rowCount = training_data.length;
        
        int input_size = 0;
        int output_size = 0;
        for(int i = 0; i < ColumnIOMap.length; i++)
        {
            if(ColumnIOMap[i] == 1)
                input_size++;
            
            if(ColumnIOMap[i] == -1)
                output_size++;
        }
        
        double[][] predictions = new double[rowCount][output_size];
        
        // Run through data set / table
        for(int r = 0; r < rowCount; r++)
        {
            // Build stimulation and response variable arrays
            double[] stimlation = new double[input_size];
            double[] response = new double[output_size];
            double[] errors = new double[output_size];
            int inputIdx = 0;
            int outputIdx = 0;

            for(int p = 0; p < ColumnIOMap.length; p++)
            {
                if(ColumnIOMap[p] == 1)
                {
                    stimlation[inputIdx] = training_data[r][p];
                    inputIdx++;
                }
                else
                {
                    response[outputIdx] = training_data[r][p];
                    outputIdx++;
                }
            }

            // Feed stimulus to learning machine
            double[] prediction = FeedForward(stimlation);
            

            // Calc cost
            for (outputIdx = 0; outputIdx < output_size; outputIdx++) 
            {
                predictions[r][outputIdx] = prediction[outputIdx];
            }

        }

        return predictions;
        
    }
    
    
}
