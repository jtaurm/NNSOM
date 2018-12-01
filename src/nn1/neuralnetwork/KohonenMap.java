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
import java.util.HashSet;
import java.util.*;
import nn1.CsvReader;
import nn1.Table;

/**
 *
 */
public class KohonenMap 
{
    ArrayList<KohonenNeuron> Neurons_list;
    
    nn1.Table TrainingData;
    
    int MapWidth;
    int MapHeight;
    
    //<editor-fold defaultstate="collapsed" desc="Initialization">
    
    public KohonenMap( nn1.Table TrainingData )
    {
        this.TrainingData = TrainingData;
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Setup">
    public void SetupHoneyCombe()
    {
        //int outputsize = (int) Math.ceil( Math.pow( (double) TrainingData.GetRowCount(), 0.5d ) * 5d );
        //int inputwidth = TrainingData.GetTableWidth();
        
        double size = Math.pow( TrainingData.GetRowCount() * TrainingData.GetTableWidth(), 0.5d ) * 5d;
        
        int outputsize = (int) Math.ceil( Math.pow( size, 0.5d ) );
        int inputwidth = (int) Math.ceil( Math.pow( size, 0.5d ) );
        
        SetupHoneyCombe( inputwidth, outputsize );
    }
    
    public void SetupHoneyCombe(int width, int height)
    {
        /*
        + + + + + + + + + +
         + + + + + + + + + +
        + + + + + + + + + +
         + + + + + + + + + +
        */
        
        MapWidth = width;
        MapHeight = height;
        
        KohonenNeuron[][] Neurons_grid = new KohonenNeuron[width][height];
        Neurons_list = new ArrayList<>();
        
        int tableWidth = TrainingData.GetTableWidth();
        int rowCount = TrainingData.GetRowCount();
        double[] x = new double[ tableWidth ];
        
        int neuronCount = 0;
        
        for(int h = 0; h < height; h++)
        {
            for(int w = 0; w < width; w++)
            {
                // Get random row sample for weight initialization
                int r = (int) (Math.random() * (double) rowCount);
                
                for(int c = 0; c < tableWidth; c++)
                {
                    Double val;
                    //val = TrainingData.GetCellValueNormalized(c, r) * (Math.random() + 0.5);
                    //val = ((double)w / (double)width) * ((double)h / (double)height);
                    val = 0.5d * (Math.random() + 0.5);
                    x[c] = (val.isNaN() ? Math.random() : val );
                }
                
                // Create neuron
                //KohonenNeuron newNeuron = new KohonenNeuron( tableWidth, "n" + h + "," + w + "_i" + r);
                KohonenNeuron newNeuron = new KohonenNeuron( tableWidth, x, "n" + h + "," + w + "_i" + r);
                
                // Insert neuron
                Neurons_grid[w][h] = newNeuron;
                Neurons_list.add(newNeuron);
                
                if(h % 2 == 0) // even layers
                {
                    if(h - 1 >= 0) // connect north
                        newNeuron.ConnectTo(Neurons_grid[w][h - 1]);
                    
                    if(h - 1 >= 0 && w - 1 >= 0) // connect north west
                        newNeuron.ConnectTo(Neurons_grid[w - 1][h - 1]);
                    
                    if(w - 1 >= 0) // connect west
                        newNeuron.ConnectTo(Neurons_grid[w - 1][h]);
                    
                    if(w == 0 && h - 1 >= 0) // connect northwest at edge
                        newNeuron.ConnectTo(Neurons_grid[width - 1][h - 1]);
                }
                else // uneven layers
                {
                    if(h - 1 >= 0) // connect north
                        newNeuron.ConnectTo(Neurons_grid[w][h - 1]);
                    
                    if(h - 1 >= 0 && w + 1 < width) // connect north east
                        newNeuron.ConnectTo(Neurons_grid[w + 1][h - 1]);
                    else if(w + 1 == width && h - 1 >= 0)
                        newNeuron.ConnectTo(Neurons_grid[0][h - 1]);
                    
                    if(w - 1 >= 0) // connect west
                        newNeuron.ConnectTo(Neurons_grid[w - 1][h]);
                }
                
                if(w + 1 == width)
                {
                    // connect east to start
                    newNeuron.ConnectTo(Neurons_grid[0][h]);
                }
                
                neuronCount++;
            }
        }
        
        System.out.println("Created SOM w " + neuronCount + " neurons.");
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Learning">
    private KohonenNeuron FindBestMatchingUnit(Double[] x)
    {
        KohonenNeuron bmu = Neurons_list.get(0);
        double bmu_dist = bmu.MeasureDistance(x);
                
        for(int i = 0; i < Neurons_list.size(); i++)
        {
            double dist = Neurons_list.get(i).MeasureDistance(x);
                
            if(dist < bmu_dist)
            {
                bmu = Neurons_list.get(i);
                bmu_dist = dist;

            }
        }
        
        return bmu;
    }
    
    private double Neighbourhood( KohonenNeuron c, KohonenNeuron n, double r )
    {
        return Math.pow( Math.E, - Math.pow( c.MeasureDistance(n), 2 ) / ( 2 * r ) );
    }
    
    public double Train_Auto( boolean verbose )
    {
        double learning_rate = 0.08;
        double mse_delta_log_threshold = 3.0d;
        int coop_radius = 3;
        double rowCount_rate = 0.9;
        
        int duration_avg = 4;
        
        return Train_Auto( learning_rate, coop_radius, rowCount_rate, mse_delta_log_threshold, duration_avg, verbose);
    }
    
    public double Train_Auto( double learning_rate, int coop_radius, double rowCount_rate, double mse_delta_log_threshold, int duration_avg, boolean verbose )
    {
        NumberFormat format7d = new DecimalFormat("#0.0000000");
        
        int rowCount = TrainingData.GetRowCount();
        int rowCount_train = (int) ((double) rowCount * rowCount_rate);
        int trainCount = 0;
        
        Double mse_avg_log = 0d;
        Double mse_slope_log = 0d;
    
        Double mse_delta_log = 0d;
        LinkedList<Double> mse_last = new LinkedList<>();
        double mse_final = 0d;
        
        while( coop_radius >= 0)
        {
            mse_last.add( Train(1, rowCount_train, learning_rate, coop_radius ) );
            trainCount++;
            
            if(mse_last.size() > duration_avg)
            {
                mse_last.removeFirst();
                
                mse_slope_log = 0d;
                mse_avg_log = 0d;
                for( int m = 0; m < duration_avg - 1; m++)
                {
                    mse_slope_log += mse_last.get(m) - mse_last.get(m+1);
                    mse_avg_log += mse_last.get(m);
                }
                mse_slope_log = mse_slope_log / duration_avg;
                mse_slope_log = Math.log10(mse_slope_log);
                mse_avg_log = mse_avg_log / duration_avg;
                mse_avg_log = Math.log10(mse_avg_log);
                mse_delta_log = mse_avg_log - mse_slope_log;
                
                if(mse_delta_log > mse_delta_log_threshold && trainCount > 20)
                {
                    if(verbose) System.out.println("\tCoop radius: " + coop_radius + " Iterations: " + trainCount + "\tError " + format7d.format( Math.pow( 10, mse_avg_log ) ) );
                    
                    coop_radius--;
                    mse_final = mse_last.getLast();
                    mse_last.clear();
                    mse_slope_log = Double.NaN;
                    mse_avg_log = Double.NaN;
                    trainCount = 0;
                    
                    
                }
            }
        }
        
        return mse_final;
    }
   
    public double Train(int repCount, int duration, double learningRate, int coop_radius )
    {
        int TableWidth = TrainingData.GetTableWidth();
        int rowCount = TrainingData.GetRowCount();
        
        HashSet<KohonenNeuron> neurons_updated = new HashSet();
        HashSet<KohonenNeuron> neurons_neighbours_r = new HashSet();
        HashSet<KohonenNeuron> neurons_queued = new HashSet();
        
        double training_error = 0;
        int training_count = 0;
        
        for(int rep = 0; rep < repCount; rep++)
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
                    
            // Iterate over time - just indices
            for(int t = 0; t < duration; t++)
            {
                // Get sample row
                int r = idx_order[t];
                Double[] x = new Double[ TableWidth ];
                for(int c = 0; c < TableWidth; c++)
                    x[c] = TrainingData.GetCellValueNormalized(c, r);
                
                // Calc local learning rate
                double learningRate_t = learningRate * (1 - (t / duration) );
                
                // Find best matching unit
                KohonenNeuron bmu = FindBestMatchingUnit(x);
                bmu.Hits++;
                
                // Update BMU - neighbourhood distance is 0
                bmu.Update(learningRate_t, 1d, x);
                
                neurons_updated.clear();
                neurons_updated.add(bmu);
                
                neurons_queued.clear();
                neurons_queued.addAll(bmu.Neighbours);
                  
                // Update BMU neighbourhood - radius 1
                
                if( coop_radius > 0)
                {
                    neurons_neighbours_r.clear();
                    neurons_neighbours_r.addAll(neurons_queued);
                    neurons_neighbours_r.removeAll(neurons_updated);
                    neurons_queued.clear();

                    neurons_neighbours_r.forEach((n1) -> {
                        double n_r1 = Neighbourhood(bmu, n1, 1);
                        n1.Update(learningRate_t, n_r1, x);

                        neurons_updated.add( n1 );
                        neurons_queued.addAll( n1.Neighbours );
                    });
                }
                if( coop_radius > 1)
                {
                    neurons_neighbours_r.clear();
                    neurons_neighbours_r.addAll(neurons_queued);
                    neurons_neighbours_r.removeAll(neurons_updated);
                    neurons_queued.clear();

                    // Update BMU neighbourhood - radius 2
                    neurons_neighbours_r.forEach((n2) -> {
                        double n_r2 = Neighbourhood(bmu, n2, 2);
                        n2.Update(learningRate_t, n_r2, x);

                        neurons_updated.add( n2 );
                        neurons_queued.addAll( n2.Neighbours );
                    });
                }
                if( coop_radius > 2)
                {
                    neurons_neighbours_r.clear();
                    neurons_neighbours_r.addAll(neurons_queued);
                    neurons_neighbours_r.removeAll(neurons_updated);
                    neurons_queued.clear();

                    // Update BMU neighbourhood - radius 3
                    neurons_neighbours_r.forEach((n3) -> {
                        double n_r3 = Neighbourhood(bmu, n3, 3);
                        n3.Update(learningRate_t, n_r3, x);
                        
                        neurons_updated.add( n3 );
                        neurons_queued.addAll( n3.Neighbours );

                    });
                }
                if( coop_radius > 3)
                {
                    neurons_neighbours_r.clear();
                    neurons_neighbours_r.addAll(neurons_queued);
                    neurons_neighbours_r.removeAll(neurons_updated);
                    neurons_queued.clear();

                    // Update BMU neighbourhood - radius 3
                    neurons_neighbours_r.forEach((n4) -> {
                        double n_r4 = Neighbourhood(bmu, n4, 4);
                        n4.Update(learningRate_t, n_r4, x);

                    });
                }
                 
                // Collect error
                double errorSum = 0d;
                int errorCount = 0;
                double[] weights = bmu.GetWeights();
                
                for(int c = 0; c < TableWidth; c++)
                    if(!x[c].isNaN())
                    {
                        errorSum += (weights[c] - x[c])*(weights[c] - x[c]);
                        errorCount++;
                    }
            
                training_error += errorSum;
                training_count += errorCount;
            }
            
            
            
            //if( (rep) % 5 == 4)
            //    System.out.println("\tCoop: " + coop_radius + " lr: " + learningRate + "\trep " + (rep + 1) + ",\tError " + (errorSum / errorCount));
            
        }
        
        return training_error / training_count;
        
    }
    
    public void ResetHits()
    {
        for(int i = 0; i < Neurons_list.size(); i++)
        {
            this.Neurons_list.get(i).Hits = 0;
        }
    }
    
    public void ClearUnused()
    {
        for(int i = Neurons_list.size() - 1; i > 0; i--)
        {
            if(this.Neurons_list.get(i).Hits == 0)
                Remove(i);
        }
    }
    
    private void Remove(int r_idx)
    {
        KohonenNeuron r = Neurons_list.get(r_idx);
        
        for(int i = 0; i < r.Neighbours.size(); i++)
        {
            KohonenNeuron n = r.Neighbours.get(i);
            n.Disconnect(r);
        }
        
        r.Neighbours.clear();
        
        Neurons_list.remove(r_idx);
    }
    
    public int GetNeuronCount()
    {
        return Neurons_list.size();
    }
    
    //</editor-fold>
    
    public double[] Predict(int row)
    {
        // Get sample row
        int TableWidth = TrainingData.GetTableWidth();
        
        Double[] x = new Double[ TableWidth ];
        for(int c = 0; c < TableWidth; c++)
            x[c] = TrainingData.GetCellValueNormalized(c, row);
        
        KohonenNeuron n = FindBestMatchingUnit(x);
        
        return n.GetWeights();
    }
    
    public nn1.Table FillMissingValues()
    {
        int rowCount = TrainingData.GetRowCount();
        int TableWidth = TrainingData.GetTableWidth();
        
        Double[] x = new Double[ TableWidth ];
        double[] y;

        // Iterate over time - just indices
        for(int r = 0; r < rowCount; r++)
        {
            // Get sample row
            for(int c = 0; c < TableWidth; c++)
                x[c] = TrainingData.GetCellValueNormalized(c, r);

            // Look for missing values
            boolean containsMissingValue = false;
            for(int c = 0; c < TableWidth; c++)
                if(x[c].isNaN())
                    containsMissingValue = true;
            
            if(!containsMissingValue)
                continue;
            
            // Predict
            y = Predict(r);
            
            // Set missing value
            for(int c = 0; c < TableWidth; c++)
                if(x[c].isNaN())
                {
                    double denorm_value = TrainingData.DenormalizeColumnValue( c, y[c] );
                    TrainingData.SetCellValue( c, r, denorm_value );
                }

        }
        
        return TrainingData;
    }
    
    public String[] PrintDimension(int position1, int position2 )
    {
        String[] lines = new String[ Neurons_list.size() + 1 ];
        
        // Header
        StringBuilder line = new StringBuilder();
        
        line.append( "\"" );
        line.append( TrainingData.GetColumnNameByPosition(position1) );
        line.append( "\",\"" );
        line.append( TrainingData.GetColumnNameByPosition(position2) );
        line.append( "\",Hits" );
        
        lines[0] = line.toString();
        
        // Data
        int neuronCount = 0;
        KohonenNeuron n;
        
        for(int i = 0; i < Neurons_list.size(); i++)
        {
                
            n = Neurons_list.get(i);
                
            line = new StringBuilder();

            line.append(n.Weights[position1] );
            line.append( "," );
            line.append(n.Weights[position2] );
            line.append( "," );
            line.append(n.Hits );

            lines[neuronCount + 1] = line.toString();
            neuronCount++;  
        }
        
        return lines;
    }
    
}

