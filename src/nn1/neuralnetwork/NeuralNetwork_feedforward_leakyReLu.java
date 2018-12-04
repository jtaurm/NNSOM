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
    
    int Propagations;
  
    double[] Neuron_input_bias;
    double[] Neuron_input_activity;
    double[] Neuron_input_descent;
    double[] Neuron_input_sum_d;
    
    double[][] Axon_input_weights;
    double[][] Axon_input_activity;
    double[][] Axon_input_descent;
    double[][] Axon_input_sum_d;
    
    double[][] Neuron_matrix_bias;
    double[][] Neuron_matrix_activity;
    double[][] Neuron_matrix_descent;
    double[][] Neuron_matrix_sum_d;
    
    double[][][] Axon_matrix_weights;
    double[][][] Axon_matrix_activity;
    double[][][] Axon_matrix_descent;
    double[][][] Axon_matrix_sum_d;
    
    double[] Neuron_output_bias;
    double[] Neuron_output_activity;
    double[] Neuron_output_descent;
    double[] Neuron_output_sum_d;
    
    double[][] Axon_output_weights;
    double[][] Axon_output_activity;
    double[][] Axon_output_descent;
    double[][] Axon_output_sum_d;
    
    public void Initialize( int inputs, int outputs, int height, int depth)
    {
        Input_height = inputs;
        Matrix_height = height;
        Matrix_depth = depth;
        Output_height = outputs;
        
        double axon_weight_scaling;
        
        // Setup inputs
        Neuron_input_bias = InitializeRandom(inputs, false, 1d);
        Neuron_input_activity = InitializeRandom(inputs, false, 1d);
        Neuron_input_descent = InitializeRandom(inputs, false, 1d);
        Neuron_input_sum_d = new double[inputs];
        
        axon_weight_scaling = 1d / (double) inputs;
        
        Axon_input_weights = InitializeRandom(inputs, height, true, axon_weight_scaling);
        Axon_input_activity = InitializeRandom(inputs, height, true, axon_weight_scaling);
        Axon_input_descent = InitializeRandom(inputs, height, true, axon_weight_scaling);
        Axon_input_sum_d = new double[inputs][height];
        
        // Setup matrix
        Neuron_matrix_bias = InitializeRandom(depth, height, false, 1d);
        Neuron_matrix_activity = InitializeRandom(depth, height, false, 1d);
        Neuron_matrix_descent = InitializeRandom(depth, height, false, 1d);
        Neuron_matrix_sum_d = new double[depth][height];
        
        axon_weight_scaling = 1d / (double) height;
        
        Axon_matrix_weights = InitializeRandom(depth-1, height, height, true, axon_weight_scaling);
        Axon_matrix_activity = InitializeRandom(depth-1, height, height, true, axon_weight_scaling);
        Axon_matrix_descent = InitializeRandom(depth-1, height, height, true, axon_weight_scaling);
        Axon_matrix_sum_d = new double[depth-1][height][height];
        
        // Setup output
        Neuron_output_bias = InitializeRandom(outputs, false, 1d);
        Neuron_output_activity = InitializeRandom(outputs, false, 1d);
        Neuron_output_descent = InitializeRandom(outputs, false, 1d);
        Neuron_output_sum_d = new double[outputs];
        
        Axon_output_weights = InitializeRandom(outputs, height, true, axon_weight_scaling);
        Axon_output_activity = InitializeRandom(outputs, height, true, axon_weight_scaling);
        Axon_output_descent = InitializeRandom(outputs, height, true, axon_weight_scaling);
        Axon_output_sum_d = new double[outputs][height];
        
        ResetSum();
    }
    
    private double[] InitializeRandom( int size, boolean only_positive, double scaling )
    {
        double[] rand_arr = new double[size];
        for(int i = 0; i < size; i++)
        {
            double r = Math.random();
            if(only_positive)
                rand_arr[i] = (r + 0.5) * scaling;
            else
                rand_arr[i] = (r > 0.5 ? r - 0.25 : -0.25 - r);
        }
        return rand_arr;
    }
    
    private double[][] InitializeRandom( int size0, int size1, boolean only_positive, double scaling )
    {
        double[][] rand_arr = new double[size0][size1];
        for(int i = 0; i < size0; i++)
        {
            rand_arr[i] = InitializeRandom(size1, only_positive, scaling);
        }
        return rand_arr;
    }
    
    private double[][][] InitializeRandom( int size0, int size1, int size2, boolean only_positive, double scaling )
    {
        double[][][] rand_arr = new double[size0][size1][size2];
        for(int i = 0; i < size0; i++)
        {
            rand_arr[i] = InitializeRandom(size1, size2, only_positive, scaling);
        }
        return rand_arr;
    }
    
    public double[] FeedForward(double[] input_stimulus)
    {
        // Stimulate inputs neurons
        for(int i = 0; i < Input_height; i++)
        {
            Neuron_input_activity[i] = Neuron_input_bias[i] + input_stimulus[i];
        }
        
        // Stimulate input axons
        for(int i = 0; i < Input_height; i++)
            for(int h = 0; h < Matrix_height; h++)
                Axon_input_activity[i][h] = Neuron_input_activity[i] * Axon_input_weights[i][h];
        
        // Stimulate first layer of the matrix neurons
        for(int h = 0; h < Matrix_height; h++)
        {
            // Set base activity eq to bias
            Neuron_matrix_activity[0][h] = Neuron_matrix_bias[0][h];
            
            // Stimulate with input axon activity
            for(int i = 0; i < Input_height; i++) // Sum over all input
                Neuron_matrix_activity[0][h] += Axon_input_activity[i][h];
            // Simulate Leaky ReLu
            for(int i = 0; i < Input_height; i++)
                if(Neuron_matrix_activity[0][h] < 0d)
                    Neuron_matrix_activity[0][h] *= 0.1;
        }
        
        // Stimulate hidden layers
        for(int d = 1; d < Matrix_depth; d++)
        {
            // Axons
            for(int h_n = 0; h_n < Matrix_height; h_n++)
            {
                // Stimulate axons from prev layer
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_activity[d-1][h_n][h_a] = Neuron_matrix_activity[d-1][h_n] * Axon_matrix_weights[d-1][h_n][h_a];
            }
            
            // Neurons
            for(int h_n = 0; h_n < Matrix_height; h_n++)
            {
                // Reset activity to bias
                Neuron_matrix_activity[d][h_n] = Neuron_matrix_bias[d][h_n];
                // Stimulate neurons from axons
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Neuron_matrix_activity[d][h_n] += Axon_matrix_activity[d-1][h_a][h_n];
                
                // Simulate Leaky ReLu
                if(Neuron_matrix_activity[d][h_n] < 0d)
                    Neuron_matrix_activity[d][h_n] *= 0.1;
            }
        }
        
        // Stimulate output layer axons
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_output_activity[h_o][h_n] = Neuron_matrix_activity[Matrix_depth - 1][h_n] * Axon_output_weights[h_o][h_n];
        
        // Stimulate output layer neurons
        for(int h_o = 0; h_o < Output_height; h_o++)
        {
            // Reset activity to bias
            Neuron_output_activity[h_o] = Neuron_output_bias[h_o];
            // Stimulate activity
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Neuron_output_activity[h_o] += Axon_output_activity[h_o][h_n];
            // No Leaky ReLu here, output neurons are linear
        }
        
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
                if(Neuron_matrix_activity[Matrix_depth - 1][h_n] < 0d)
                Neuron_matrix_descent[Matrix_depth - 1][h_n] *= 0.1;
            
        }
        
        // Hidden layers
        for(int d = Matrix_depth - 2; d >= 0; d--)
        {
            for(int h_n = 0; h_n < Matrix_height; h_n++)
            {
                // Calculate Axon dZ = from.activity * to.dZ
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_descent[d][h_n][h_a] = Neuron_matrix_activity[d][h_n] * Neuron_matrix_descent[d + 1][h_a];
                
                // Calculate Neuron dZ = axon_out[].weight * neuron_out[].dZ
                Neuron_matrix_descent[d][h_n] = 0d;
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Neuron_matrix_descent[d][h_n] += Axon_matrix_weights[d][h_n][h_a] * Neuron_matrix_descent[d + 1][h_a];
                
                // Simulate Leaky ReLu
                for(int h_o = 0; h_o < Output_height; h_o++)
                    if(Neuron_matrix_activity[d][h_n] < 0d)
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
            if(Neuron_input_activity[h_i] < 0d)
                Neuron_input_descent[h_i] *= 0.1;
        }
        
        // Reset propagation count
        Propagations++;
        
        // Add descents to acc sum
        for(int h_o = 0; h_o < Output_height; h_o++)
            Neuron_output_sum_d[h_o] += Neuron_output_descent[h_o];
        
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_output_sum_d[h_o][h_n] += Axon_output_descent[h_o][h_n];
        
        for(int d = 0; d < Matrix_depth - 1; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_sum_d[d][h_n][h_a] += Axon_matrix_descent[d][h_n][h_a];
        
        for(int d = 0; d < Matrix_depth; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Neuron_matrix_sum_d[d][h_n] += Neuron_matrix_descent[d][h_n];
        
        for(int h_i = 0; h_i < Input_height; h_i++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_input_sum_d[h_i][h_n] += Axon_input_descent[h_i][h_n];
        
        for(int h_i = 0; h_i < Input_height; h_i++)
            Neuron_input_sum_d[h_i] += Neuron_input_descent[h_i];
    }
    
    private void ResetSum()
    {
        Propagations = 0;
        
        // reset acc sum
        for(int h_o = 0; h_o < Output_height; h_o++)
            Neuron_output_sum_d[h_o] = 0d;
        
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_output_sum_d[h_o][h_n] = 0d;
        
        for(int d = 0; d < Matrix_depth - 1; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_sum_d[d][h_n][h_a] = 0d;
        
        for(int d = 0; d < Matrix_depth; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Neuron_matrix_sum_d[d][h_n] = 0d;
        
        for(int h_i = 0; h_i < Input_height; h_i++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_input_sum_d[h_i][h_n] = 0d;
        
        for(int h_i = 0; h_i < Input_height; h_i++)
            Neuron_input_sum_d[h_i] = 0d;
    }
    
    public void UpdateWeights_Acc(double learning_rate)
    {
        // Update output
        for(int h_o = 0; h_o < Output_height; h_o++)
            Neuron_output_bias[h_o] += Neuron_output_sum_d[h_o] * learning_rate * (1d/(double)Propagations);
        
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_output_weights[h_o][h_n] += Axon_output_sum_d[h_o][h_n] * learning_rate * (1d/(double)Propagations);
        
        // Update hidden layers
        for(int d = 0; d < Matrix_depth - 1; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_weights[d][h_n][h_a] += Axon_matrix_sum_d[d][h_n][h_a] * learning_rate * (1d/(double)Propagations);
        
        for(int d = 0; d < Matrix_depth; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Neuron_matrix_bias[d][h_n] += Neuron_matrix_sum_d[d][h_n] * learning_rate * (1d/(double)Propagations);

        // Update output
        for(int h_i = 0; h_i < Input_height; h_i++)
            Neuron_input_bias[h_i] += Neuron_input_sum_d[h_i] * learning_rate * (1d/(double)Propagations);
        
        for(int h_i = 0; h_i < Input_height; h_i++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_input_weights[h_i][h_n] += Axon_input_sum_d[h_i][h_n] * learning_rate * (1d/(double)Propagations);
        
        ResetSum();
    }
    
    public void UpdateWeights(double learning_rate)
    {
        // Update output
        for(int h_o = 0; h_o < Output_height; h_o++)
            Neuron_output_bias[h_o] += Neuron_output_descent[h_o] * learning_rate;
        
        for(int h_o = 0; h_o < Output_height; h_o++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_output_weights[h_o][h_n] += Axon_output_descent[h_o][h_n] * learning_rate;
        
        // Update hidden layers
        for(int d = 0; d < Matrix_depth - 1; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                for(int h_a = 0; h_a < Matrix_height; h_a++)
                    Axon_matrix_weights[d][h_n][h_a] += Axon_matrix_descent[d][h_n][h_a] * learning_rate;
        
        for(int d = 0; d < Matrix_depth; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Neuron_matrix_bias[d][h_n] += Neuron_matrix_descent[d][h_n] * learning_rate;

        // Update output
        for(int h_i = 0; h_i < Input_height; h_i++)
            Neuron_input_bias[h_i] += Neuron_input_descent[h_i] * learning_rate;
        
        for(int h_i = 0; h_i < Input_height; h_i++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                Axon_input_weights[h_i][h_n] += Axon_input_descent[h_i][h_n] * learning_rate;
    }
    
    public double[] Train(double[][] training_data, int[] ColumnIOMap, double learning_rate, boolean monte_carlo, boolean batch_mode)
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
        double[] eSum = new double[output_size];
        
        // Generate indices list
        int[] idx_order = new int[rowCount];
        for(int i = 0; i < rowCount; i++)
            idx_order[i] = i;

        // Randomize indices list
        if(monte_carlo)
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
            double[] costs = new double[output_size];
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

            // Calc cost / error
            for (outputIdx = 0; outputIdx < output_size; outputIdx++) 
            {
                errors[outputIdx] = response[outputIdx] - predictions[outputIdx];
                costs[outputIdx] = 2 * errors[outputIdx];
            }
            Errors.add(errors);

            // Back propagate costs
            BackPropagate(costs);

            // Update weights
            if(!batch_mode)
                UpdateWeights(learning_rate);
            //System.out.println( predictions[0] + "\terr: "+ errors[0] + "\tO0 bias: " + Neuron_output_bias[0] + "\tdZ: " + Neuron_output_descent[0] + "\t3,10 bias: " + Neuron_matrix_bias[3][10] + "\tdZ: " + Neuron_matrix_descent[3][10] );

        }
        
        // Update weights
        if(batch_mode)
            UpdateWeights_Acc(learning_rate);
        else
            ResetSum();

        // Calc training MSE
        Arrays.fill(eSum, 0d);

        for(int i = 0; i < Errors.size(); i++)
        {
            for (int outputIdx = 0; outputIdx < output_size; outputIdx++) 
                eSum[outputIdx] += Math.pow( Errors.get(i)[outputIdx], 2d );
        }

        for (int outputIdx = 0; outputIdx < output_size; outputIdx++) 
            eSum[outputIdx] = eSum[outputIdx] / Errors.size();

        Errors.clear();
        
        return eSum;
        
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
    
    
    public int Count0Neurons()
    {
        int dead = 0;
        
        for(int d = 0; d < Matrix_depth - 1; d++)
            for(int h_n = 0; h_n < Matrix_height; h_n++)
                if( Neuron_matrix_activity[d][h_n] < 1E-5)
                    dead++;
        
        return dead;
    }
    
}
