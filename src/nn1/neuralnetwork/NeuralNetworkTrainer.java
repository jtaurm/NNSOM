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

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;
import nn1.Table;

/**
 *
 */
public class NeuralNetworkTrainer 
{
    nn1.Table TrainingData;
    
    int[] ColumnIOMap;
    
    int[] InputIdx;
    int[] OutputIdx;
    
    int InputSize;
    int OutputSize;
    
    NeuralNetwork LearningMachine;
    
    public NeuralNetworkTrainer(NeuralNetwork machine)
    {
        LearningMachine = machine;
        InputSize = 0;
        OutputSize = 0;
    }
    
    public void SetTrainingData(nn1.Table trainingData, int[] columnIOMap)
    {
        TrainingData = trainingData;
        ColumnIOMap = columnIOMap;
                
        for(int i = 0; i < columnIOMap.length; i++)
        {
            if(columnIOMap[i] == 1 )
            {
                InputSize++;
                continue;
            }
            
            if(columnIOMap[i] == -1 )
            {
                OutputSize++;
            }
        }
        
        InputIdx = new int[InputSize];
        OutputIdx = new int[OutputSize];
        
        int inputIdx = 0;
        int outputIdx = 0;
        
        for(int i = 0; i < columnIOMap.length; i++)
        {
            if(columnIOMap[i] == 1 )
            {
                InputIdx[inputIdx] = i;
                inputIdx++;
                continue;
            }
            
            if(columnIOMap[i] == -1 )
            {
                OutputIdx[outputIdx] = i;
                outputIdx++;
            }
        }
        
    }
    
    public void Train(int repCount, double learningRate)
    {
        int rowCount = TrainingData.GetRowCount();
        ArrayList<Neuron> OutputNeurons = LearningMachine.GetOutputNeurons();
        
        ArrayList<Double[]> Errors = new ArrayList<>();
        ArrayList<Double[]> Predictions = new ArrayList<>();
        
        // for each repeatition of data set
        for(int reps = 0; reps < repCount; reps++)
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
                Double[] stimlation = new Double[InputSize];
                Double[] response = new Double[OutputSize];
                Double[] errors = new Double[OutputSize];
                Double[] predictions = new Double[OutputSize];
                int inputIdx = 0;
                int outputIdx = 0;

                for(int p = 0; p < TrainingData.GetTableWidth(); p++)
                {
                    if(ColumnIOMap[p] == 1)
                    {
                        stimlation[inputIdx] = TrainingData.GetCellValueNormalized(p, idx_order[r] );
                        inputIdx++;
                    }
                    else
                    {
                        response[outputIdx] = TrainingData.GetCellValueNormalized(p, idx_order[r] );
                        outputIdx++;
                    }
                }

                // Feed stimulus to learning machine
                LearningMachine.FeedInputs(stimlation);

                // Collect neuron output (predictions) and calc cost
                for (outputIdx = 0; outputIdx < OutputIdx.length; outputIdx++) 
                {
                    Neuron o = OutputNeurons.get(outputIdx);

                    if( outputIdx == 1 && response[outputIdx].isNaN() ) // Fix NaN for exalted shards
                        response[1] = response[0];
                    
                    // Collect activity
                    predictions[outputIdx] = o.GetActivity();
                    
                    // Calculate cost
                    errors[outputIdx] = response[outputIdx] - o.GetActivity();
                }

                Errors.add(errors);
                Predictions.add(predictions);

                LearningMachine.PropagateError( errors, learningRate );

            }
            
            // Calc training MSE
            Double eSum = 0d;
            int eCount = 0;
            for(Double[] e: Errors)
            {
                if(e[0].isNaN())
                    continue;
                
                eSum += e[0] * e[0];
                eCount++;
            }
            eSum = eSum / eCount;
            
            Errors.clear();
            Predictions.clear();
            
            if (reps % 50 == 0)
                System.out.println( "\tlr: " + learningRate+ "\trep: " + reps + "\ttraining MSE: " + eSum );
            
        }
    }
    
    public void Train_k(int repCount, double learningRate, int k)
    {
        int rowCount = TrainingData.GetRowCount();
        ArrayList<Neuron> OutputNeurons = LearningMachine.GetOutputNeurons();
        
        ArrayList<Double[]> TrainingErrors = new ArrayList<>();
        ArrayList<Double[]> TestErrors = new ArrayList<>();
        
        ArrayList<Double[]> Predictions = new ArrayList<>();
        
        int k_amount = Math.round( rowCount / k );
        
        // for each repeatition of data set
        for(int reps = 0; reps < repCount; reps++)
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
                Double[] stimlation = new Double[InputSize];
                Double[] response = new Double[OutputSize];
                Double[] errors = new Double[OutputSize];
                Double[] predictions = new Double[OutputSize];
                int inputIdx = 0;
                int outputIdx = 0;

                for(int p = 0; p < TrainingData.GetTableWidth(); p++)
                {
                    if(ColumnIOMap[p] == 1)
                    {
                        stimlation[inputIdx] = TrainingData.GetCellValueNormalized(p, idx_order[r] );
                        inputIdx++;
                    }
                    else
                    {
                        response[outputIdx] = TrainingData.GetCellValueNormalized(p, idx_order[r] );
                        outputIdx++;
                    }
                }

                // Feed stimulus to learning machine
                LearningMachine.FeedInputs(stimlation);

                // Collect neuron output (predictions) and calc cost
                for (outputIdx = 0; outputIdx < OutputIdx.length; outputIdx++) 
                {
                    Neuron o = OutputNeurons.get(outputIdx);

                    if( outputIdx == 1 && response[outputIdx].isNaN() ) // Fix NaN for exalted shards
                        response[1] = response[0];
                    
                    // Collect activity
                    predictions[outputIdx] = o.GetActivity();
                    
                    // Calculate cost
                    errors[outputIdx] = response[outputIdx] - o.GetActivity();
                }

                if( r > rowCount - k_amount)
                { // Do test MSE
                    TestErrors.add(errors);
                    Predictions.add(predictions);
                }
                else
                { // Train network
                    TrainingErrors.add(errors);
                    Predictions.add(predictions);

                    LearningMachine.PropagateError( errors, learningRate );
                }
                
                

            }
            
            // Calc training MSE
            Double ErrSum_train = 0d;
            int eCount = 0;
            for(Double[] e: TrainingErrors)
            {
                if(e[0].isNaN())
                    continue;
                
                ErrSum_train += e[0] * e[0];
                eCount++;
            }
            ErrSum_train = ErrSum_train / eCount;
            
            // Calc test MSE
            Double ErrSum_test = 0d;
            eCount = 0;
            for(Double[] e: TestErrors)
            {
                if(e[0].isNaN())
                    continue;
                
                ErrSum_test += e[0] * e[0];
                eCount++;
            }
            ErrSum_test = ErrSum_test / eCount;
            
            // Clear for next run
            TrainingErrors.clear();
            TestErrors.clear();
            Predictions.clear();
            
            if (reps % 25 == 0)
                System.out.println( "\tlr: " + learningRate+ "\trep: " + reps + "\ttraining MSE: " + ErrSum_train + "\ttest MSE: " + ErrSum_test);
            
        }
    }
    
    
    public ArrayList<double[]> MapInputStrength()
    {
        // Ready return structure
        ArrayList<double[]> inputStrength = new ArrayList<>();
        
        // Map input strength
        int inputIdx = 0;
        int outputIdx = 0;
        
        Double[] stimlation = new Double[InputSize];
        
        // Feed zero input as reference
        LearningMachine.FeedInputs(stimlation);
        
        // Read zero input
        double[] zeroInputActivity = new double[OutputSize];
        for (outputIdx = 0; outputIdx < OutputSize; outputIdx++)
        {
            zeroInputActivity[outputIdx] = LearningMachine.GetOutputNeurons().get(outputIdx).GetActivity();
        }
        
        // Stimulate each input separately
        for(int p = 0; p < TrainingData.GetTableWidth(); p++)
        {
            double[] predictions = new double[OutputSize];

            Arrays.fill( stimlation, 0d);
            
            if(ColumnIOMap[p] == 1)
            {
                // Only inputs are worth stimulating
                stimlation[inputIdx] = 1d;
                inputIdx++;
                
                // Feed single input
                LearningMachine.FeedInputs(stimlation);

                // Collect output neuron activity
                for (outputIdx = 0; outputIdx < OutputSize; outputIdx++) 
                {
                    predictions[outputIdx] = LearningMachine.GetOutputNeurons().get(outputIdx).GetActivity() - zeroInputActivity[outputIdx];
                }
            }
            else
            {
                // non inputs are zero
                for (outputIdx = 0; outputIdx < OutputSize; outputIdx++) 
                {
                    predictions[outputIdx] = 0d;
                }
            }

            inputStrength.add(predictions);
            
        }
        
        return inputStrength;
    }
    
    
    public ArrayList<ArrayList<Double>> Predict()
    {
        // init return value
        ArrayList<ArrayList<Double>> predictionCols;
        predictionCols = new ArrayList<>();
        
        int outputIdx;
        for (outputIdx = 0; outputIdx < OutputSize; outputIdx++) 
        {
            predictionCols.add( new ArrayList<>() );
        }
        
        // Run through data set and collect output
        for(int r = 0; r < TrainingData.GetRowCount(); r++)
        {
            // Build stimulation array
            Double[] stimlation = new Double[InputSize];
            int inputIdx = 0;

            for(int p = 0; p < TrainingData.GetTableWidth(); p++)
            {
                if(ColumnIOMap[p] == 1)
                {
                    stimlation[inputIdx] = TrainingData.GetCellValueNormalized(p, r);
                    inputIdx++;
                }
            }

            // Feed stimulation to learning machine
            LearningMachine.FeedInputs(stimlation);
            
            // Collect output neuron activity
            for (outputIdx = 0; outputIdx < OutputSize; outputIdx++) 
            {
                predictionCols.get(outputIdx).add(TrainingData.DenormalizeColumnValue(OutputIdx[outputIdx], 
                                LearningMachine.GetOutputNeurons().get(outputIdx).GetActivity() 
                        ) 
                );
                
            }
        }
        
        return predictionCols;
    }
    
}
