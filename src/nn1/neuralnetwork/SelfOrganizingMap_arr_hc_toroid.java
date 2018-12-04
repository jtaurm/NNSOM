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
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 */
public class SelfOrganizingMap_arr_hc_toroid 
{
    //<editor-fold defaultstate="collapsed" desc="Vars -- Data">
    nn1.Table TrainingData;
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Vars -- Weights and tracking">
    
    double[][][] Neuron_matrix_weights;
    double[][][] Neuron_matrix_gradients;
    
    int Neuron_matrix_width;
    int Neuron_matrix_height;
    int Neuron_matrix_depth;
    
    int[][] Neuron_matrix_hits;
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Func -- Initialization">
    public SelfOrganizingMap_arr_hc_toroid( nn1.Table TrainingData )
    {
        this.TrainingData = TrainingData;
        this.Neuron_matrix_depth = TrainingData.GetTableWidth();
    }
    
    public void Initialize()
    {
        double size = Math.pow( TrainingData.GetRowCount() * TrainingData.GetTableWidth(), 0.5d ) * 5.0d;
        
        int outputsize = (int) Math.ceil( Math.pow( size, 0.5d ) );
        int inputwidth = (int) Math.ceil( Math.pow( size, 0.5d ) );
        
        Initialize( inputwidth, outputsize );
    }
    
    public void Initialize(int width, int height)
    {
        /*
        + + + + + + + + + +
         + + + + + + + + + +
        + + + + + + + + + +
         + + + + + + + + + +
        */
        
        Neuron_matrix_width = width;
        Neuron_matrix_height = height;
        
        Neuron_matrix_weights = new double[Neuron_matrix_width][Neuron_matrix_height][Neuron_matrix_depth];
        
        Neuron_matrix_gradients = new double[Neuron_matrix_width][Neuron_matrix_height][Neuron_matrix_depth];
        Neuron_matrix_hits = new int[Neuron_matrix_width][Neuron_matrix_height];
        
        int neuronCount = 0;
        
        ArrayList<double[]> pcs = TrainingData.GetPCs(0.001, 2);
        
        double[] pc1 = pcs.get(0);
        double[] pc2 = pcs.get(1);
        
        for(int h = 0; h < height; h++)
        {
            for(int w = 0; w < width; w++)
            {
                for(int d = 0; d < Neuron_matrix_depth; d++)
                {
                    double scalar_width = ( (double)w / ( (double) width - 1d ) );
                    double scalar_height = ( (double)h / ( (double) height - 1d ) );
                    double scalar_pc = 1 / ( Math.abs( pc1[d] ) + Math.abs( pc2[d] ) );
                    
                    Neuron_matrix_weights[w][h][d] = 
                            scalar_pc * Math.abs( pc1[d] ) * scalar_width +
                            scalar_pc * Math.abs( pc2[d] ) * scalar_height
                            ;
                    //if(d == 37 || d == 12) System.out.println("\tw: " + w + " ~" + scalar_width + "\th: " + h + "\td: " + d + "\tinit: " + Neuron_matrix_weights[w][h][d] + "\tpc1.d: " + pc1[d] + "\tpc2.d: " + pc2[d] );
                }
                
                neuronCount++;
            }
        }
        
        System.out.println("Created SOM w " + neuronCount + " neurons. Initialized from PC1 and PC2 with " + Neuron_matrix_depth + " dimension.");
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Func -- Learning">
    private int[] FindBestMatchingUnit(Double[] x)
    {
        int bmu_w = 0;
        int bmu_h = 0;
        double bmu_distance = Double.MAX_VALUE;
        
        for(int w = 0; w < Neuron_matrix_width; w++)
        {
            for(int h = 0; h < Neuron_matrix_height; h++)
            {
                if(Neuron_matrix_hits[w][h] == -1)
                    continue;
                
                double distance = 0d;

                for(int d = 0; d < Neuron_matrix_depth; d++)
                {
                    if( !x[d].isNaN() )
                        distance += Math.pow(Neuron_matrix_weights[w][h][d] - x[d], 2d );
                }

                //distance = Math.pow(distance, 0.5d);
                
                if(distance < bmu_distance)
                {
                    bmu_w = w;
                    bmu_h = h;
                    bmu_distance = distance;

                }
            }
        }
        
        return new int[] { bmu_w, bmu_h };
    }
    
    private double Neighbourhood( int c_w, int c_h, int n_w, int n_h, double r )
    {
        double distance = 0d;

        for(int d = 0; d < Neuron_matrix_depth; d++)
        {
            distance += Math.pow(Neuron_matrix_weights[c_w][c_h][d] - Neuron_matrix_weights[n_w][n_h][d], 2d );
        }
        distance = Math.pow(distance, 0.5d);
        
        return Math.pow( Math.E, - Math.pow( distance, 2 ) / ( 2 * r ) );
    }
    
    /**
     * Automatic training algorithm.
     * @param verbose Print results after each radius. (true)
     * @return 
     */
    public double Train_Auto( boolean verbose )
    {
        double learning_rate = 0.05;
        double mse_delta_log_threshold = 3.2d;
        int coop_radius = 3;
        double rowCount_rate = 0.90;
        
        int duration_avg = 10;
        int min_rad_iteration = duration_avg * 2;
        
        return Train_Auto( learning_rate, coop_radius, rowCount_rate, mse_delta_log_threshold, min_rad_iteration, duration_avg, verbose);
    }
    
    /**
     * Automatic training algorithm, with tweakable arguments.
     * @param learning_rate Weight correction rate. (0.10)
     * @param coop_radius Starting/max update radius, iteratively reduced. (2)
     * @param rowCount_rate Ratio of rows in training data to use each training session (1.00). Rows are picked in a random order, if 1.0 then all rows are used at each session.
     * @param mse_delta_log_threshold Improvement rate at which to keep training if above. In log10 scale, ie. 3.0 = 0.001 = 0.1% improvement per session or better. (3.5)
     * @param min_rad_iteration Minimum amount of iterations at each radius. (50)
     * @param duration_avg Improvement rate is calculated over an average amount of training sessions. (10)
     * @param verbose Print results after each radius. (true)
     * @return 
     */    
    public double Train_Auto( double learning_rate, int coop_radius, double rowCount_rate, double mse_delta_log_threshold, int min_rad_iteration, int duration_avg, boolean verbose )
    {
        NumberFormat format7d = new DecimalFormat("#0.0000000");
        
        Double[][] trainingData = TrainingData.GetNormalizedDataSetAsArray_cr();
        
        int rowCount = TrainingData.GetRowCount();
        int trainCount_radius = 0;
        int trainCount_total = 0;
        int trainCount_base = 1;
        
        int coop_radius_current = coop_radius;
        
        Double mse_avg_log;
        Double mse_avg;
        Double mse_slope_log;
        Double mse_slope;
    
        Double mse_delta_log;
        LinkedList<Double> mse_last = new LinkedList<>();
        double mse_final = 0d;
        
        while( coop_radius_current >= 0)
        {   // For each radius down to 0
            
            // Train
            //mse_last.add( Train_MonteCarlo(trainingData, trainCount_base, rowCount, learning_rate, coop_radius ) );
            mse_last.add( Train_SeededOrder(trainingData, trainCount_base, learning_rate, coop_radius_current, trainCount_total ) );
            
            trainCount_radius += trainCount_base;
            trainCount_total += trainCount_base;
            
            if(verbose) System.out.print("|");
            
            // Analyse results
            if(mse_last.size() > duration_avg + 1)
            {
                mse_last.removeFirst(); // Keep last [duration_avg+1] results
                
                // Calc average and slope from results
                mse_slope = 0d;
                mse_avg = 0d;
                
                for( int m = 0; m < duration_avg; m++)
                {
                    mse_slope += mse_last.get(m) - mse_last.get(m+1);
                    mse_avg += mse_last.get(m);
                }
                mse_slope = mse_slope / duration_avg;
                mse_slope_log = (mse_slope < 0 ? Double.NEGATIVE_INFINITY : Math.log10(mse_slope) );
                mse_avg = mse_avg / duration_avg;
                mse_avg_log = (mse_avg < 0 ? Double.NEGATIVE_INFINITY : Math.log10(mse_avg) );
                
                // Calc relative improvement ( = -infinity if slope is not improving )
                mse_delta_log = mse_avg_log - mse_slope_log;
                
                if( mse_delta_log > mse_delta_log_threshold && trainCount_radius > min_rad_iteration)
                {   // Improvement is too low
                    if(verbose) System.out.println();
                    if(verbose) System.out.println("\tCoop radius: " + coop_radius_current + " Iterations: " + trainCount_radius + "\tError " + format7d.format( Math.pow( 10, mse_avg_log ) ) );
                    
                    // Reduce radius
                    coop_radius_current--;
                    
                    // clear vars
                    mse_final = mse_last.getLast();
                    mse_last.clear();
                    trainCount_radius = 0;
                    
                    if(coop_radius_current == 0) // clear for last round
                        ResetHits();
                }
            }
            
        }
        
        return mse_final;
    }
    
    // ****************************************** e  nw   w  sw  se e
    private final int[] walk_r1_e_w = new int[] { 1, -1, -1,  0, 0, 1 };
    private final int[] walk_r1_e_h = new int[] { 0, -1,  0,  1, 1, 0 };
    // ****************************************** e  nw   w  sw  se e
    private final int[] walk_r1_u_w = new int[] { 1,  0, -1, -1, 1, 1 };
    private final int[] walk_r1_u_h = new int[] { 0, -1,  0,  1, 1, 0 };
    
    public double Train_MonteCarlo(Double[][] trainingData, int repCount, int duration, double learningRate, int coop_radius )
    {
        int rowCount = TrainingData.GetRowCount();
                
        double training_error = 0d;
        double test_error = 0d;
        
        int[] test_indices;
        int test_count = (int)((double)rowCount / 10);
        
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
            
            test_indices = Arrays.copyOf(idx_order, test_count);
                    
            // Train single session
            training_error += Train_SingleSession( idx_order, learningRate, coop_radius, trainingData);
            test_error += PredictionMSE(test_indices, trainingData);
            
        }
        
        return test_error / repCount;
        
    }
    
    public double Train_SeededOrder(Double[][] trainingData, int repCount, double learningRate, int coop_radius, int seed )
    {
        int row_count = TrainingData.GetRowCount();
        seed = Math.max( seed % row_count, 1 );
        
        double training_error = 0d;
        
        for(int rep = 0; rep < repCount; rep++)
        {
            // Generate indices list
            int[] idx_order = new int[row_count];
            
            int i = 0;
            for(int s = 0; s < seed; s++)
                for(int r = s; r < row_count; r += seed)
                {
                    idx_order[i] = r;
                    i++;
                }
                    
            // Train single session
            // All data is passed through - only training error is available
            training_error += Train_SingleSession( idx_order, learningRate, coop_radius, trainingData);
        }
        
        return training_error / repCount;
        
    }
    
    public double Train_SingleSession(int[] idx_order, double learningRate, int coop_radius, Double[][] trainingData )
    {
        int TableWidth = trainingData.length;
                
        double training_error = 0;
        int training_count = 0;

        // Iterate over time - just indices
        for(int t = 0; t < idx_order.length; t++)
        {
            // Get sample row
            int row = idx_order[t];
            Double[] x = new Double[ TableWidth ];
            for(int c = 0; c < TableWidth; c++)
                x[c] = trainingData[c][row];

            // Calc local learning rate
            double learningRate_t = learningRate;// * (1 - (t / idx_order.length) );

            // Find best matching unit
            int[] bmu_coords = FindBestMatchingUnit(x);
            int bmu_w = bmu_coords[0];
            int bmu_h = bmu_coords[1];

            Neuron_matrix_hits[ bmu_w ][ bmu_h ]++;

            // Update BMU - neighbourhood distance is 0

            // bmu.Update(learningRate_t, 1d, x);
            // m(i + 1) = m(i) + a(t) * Neighbourhood( c, i ) * ( x - m(i))

            double neighbourhood_dist = 1d;

            for(int d = 0; d < Neuron_matrix_depth; d++)
            {
                if( !x[d].isNaN() )
                {
                    Neuron_matrix_gradients[bmu_w][bmu_h][d] = learningRate_t * neighbourhood_dist * ( x[d] - Neuron_matrix_weights[bmu_w][bmu_h][d] );
                    Neuron_matrix_weights[bmu_w][bmu_h][d] = Neuron_matrix_weights[bmu_w][bmu_h][d] + Neuron_matrix_gradients[bmu_w][bmu_h][d];
                }
                else // If unknown, do as last time
                {
                    Neuron_matrix_weights[bmu_w][bmu_h][d] = Neuron_matrix_weights[bmu_w][bmu_h][d] + Neuron_matrix_gradients[bmu_w][bmu_h][d];
                    Neuron_matrix_gradients[bmu_w][bmu_h][d] = 0d;
                }

                //if( Double.isInfinite(Neuron_matrix_weights[bmu_w][bmu_h][d]))
                //    System.out.println("wah");
            }

            int u_h, u_w;

            for(int r = 1; r < coop_radius; r++) 
            {   // r is current radius
                u_h = bmu_h + r;
                u_w = bmu_w + r;

                for(int dir = 0; dir < 6; dir++) 
                {   // dir is current direction
                    for(int w = 1; w <= r; w++) 
                    {   // w is current walking dist

                        // walk
                        if( u_h % 2 == 0) 
                        {   // even
                            u_w = (u_w + walk_r1_e_w[dir] + Neuron_matrix_width) % Neuron_matrix_width;
                            u_h = (u_h + walk_r1_e_h[dir] + Neuron_matrix_height) % Neuron_matrix_height;
                        }
                        else 
                        {   // ueven
                            u_w = (u_w + walk_r1_u_w[dir] + Neuron_matrix_width) % Neuron_matrix_width;
                            u_h = (u_h + walk_r1_u_h[dir] + Neuron_matrix_height) % Neuron_matrix_height;
                        }

                        neighbourhood_dist = Neighbourhood( bmu_w, bmu_h, u_w, u_h, r);

                        //System.out.println( "\tr:" + r + "\tw:" + w + "\tdir:"+dir+"\tu_w:"+u_w + "\tu_h:"+u_h);
                        for(int d = 0; d < Neuron_matrix_depth; d++)
                        {
                            //System.out.println( "\tr:" + r + "\tw:" + w + "\tdir:"+dir+"\td:" + d + "\tu_w:"+u_w + "\tu_h:"+u_h);
                            if( !x[d].isNaN() )
                            {
                                Neuron_matrix_gradients[u_w][u_h][d] = learningRate_t * neighbourhood_dist * ( x[d] - Neuron_matrix_weights[u_w][u_h][d] );
                                Neuron_matrix_weights[u_w][u_h][d] = Neuron_matrix_weights[u_w][u_h][d] + Neuron_matrix_gradients[u_w][u_h][d];
                            }
                            else // If unknown, do as last time
                            {
                                Neuron_matrix_weights[u_w][u_h][d] = Neuron_matrix_weights[u_w][u_h][d] + Neuron_matrix_gradients[u_w][u_h][d];
                                Neuron_matrix_gradients[u_w][u_h][d] = 0d;
                            }
                        }

                    }

                }
            }

            // Collect BMU error
            double errorSum = 0d;
            int errorCount = 0;

            for(int d = 0; d < TableWidth; d++)
                if(!x[d].isNaN())
                {
                    errorSum += (Neuron_matrix_weights[bmu_w][bmu_h][d] - x[d])*(Neuron_matrix_weights[bmu_w][bmu_h][d] - x[d]);
                    errorCount++;
                }

            training_error += errorSum;
            training_count += errorCount;
        }
        
        return training_error / training_count;
        
    }
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Func -- Cleaning">
    private void ResetHits()
    {
        for(int w = 0; w < Neuron_matrix_width; w++)
        {
            for(int h = 0; h < Neuron_matrix_height; h++)
            {
                Neuron_matrix_hits[w][h] = 0;
            }
        }
    }
    
    public void ClearUnused()
    {
        for(int w = 0; w < Neuron_matrix_width; w++)
        {
            for(int h = 0; h < Neuron_matrix_height; h++)
            {
                if(Neuron_matrix_hits[w][h] == 0)
                    Neuron_matrix_hits[w][h] = -1;
            }
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Func -- Prediction and output">
    public int GetNeuronCount()
    {
        int neuron_count = 0;
        
        for(int w = 0; w < Neuron_matrix_width; w++)
        {
            for(int h = 0; h < Neuron_matrix_height; h++)
            {
                if(Neuron_matrix_hits[w][h] >= 0)
                    neuron_count++;
            }
        }
        return neuron_count;
    }
    
    public double PredictionMSE(int[] indices, Double[][] trainingData)
    {
        double mse = 0d;
        Double[] row_data = new Double[Neuron_matrix_depth];
        double[] row_prediction;
        for(int i = 0; i < indices.length; i++)
        {
            for(int c = 0; c < Neuron_matrix_depth; c++)
                row_data[c] = trainingData[c][i];

            row_prediction = Predict(i, trainingData);
            
            for(int c = 0; c < Neuron_matrix_depth; c++)
                if(!row_data[c].isNaN())
                    mse += Math.pow( row_prediction[c] - row_data[c], 2 );
        }
        return mse / ((double) indices.length);
    }
    
    public double[] Predict(int row, Double[][] trainingData)
    {
        // Get sample row
        Double[] x = new Double[ Neuron_matrix_depth ];
        for(int c = 0; c < Neuron_matrix_depth; c++)
            x[c] = trainingData[c][row];
        
        int[] bmu_coords = FindBestMatchingUnit(x);
        
        return Neuron_matrix_weights[ bmu_coords[0] ][ bmu_coords[1] ];
    }
    
    public double[] Predict(int row)
    {
        // Get sample row
        Double[] x = new Double[ Neuron_matrix_depth ];
        for(int c = 0; c < Neuron_matrix_depth; c++)
            x[c] = TrainingData.GetCellValueNormalized(c, row);
        
        int[] bmu_coords = FindBestMatchingUnit(x);
        
        return Neuron_matrix_weights[ bmu_coords[0] ][ bmu_coords[1] ];
    }
    
    public nn1.Table FillMissingValues()
    {
        int rowCount = TrainingData.GetRowCount();
        
        Double[] x = new Double[ Neuron_matrix_depth ];
        double[] y;

        // Iterate over time - just indices
        for(int r = 0; r < rowCount; r++)
        {
            // Get sample row
            for(int c = 0; c < Neuron_matrix_depth; c++)
                x[c] = TrainingData.GetCellValueNormalized(c, r);

            // Look for missing values
            boolean containsMissingValue = false;
            for(int c = 0; c < Neuron_matrix_depth; c++)
                if(x[c].isNaN())
                    containsMissingValue = true;
            
            if(!containsMissingValue)
                continue;
            
            // Predict
            y = Predict(r);
            
            // Set missing value
            for(int c = 0; c < Neuron_matrix_depth; c++)
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
        int neuron_count = GetNeuronCount();
        String[] lines = new String[ neuron_count + 1 ];
        
        // Header
        StringBuilder line = new StringBuilder();
        
        line.append( "\"" );
        line.append( TrainingData.GetColumnNameByPosition(position1) );
        line.append( "\",\"" );
        line.append( TrainingData.GetColumnNameByPosition(position2) );
        line.append( "\",Hits,x,y" );
        
        lines[0] = line.toString();
        
        // Data
        int neuronCount = 0;
        
        for(int w = 0; w < Neuron_matrix_width; w++)
        {
            for(int h = 0; h < Neuron_matrix_height; h++)
            {
                if( Neuron_matrix_hits[w][h] < 0)
                    continue;
                
                line = new StringBuilder();

                line.append( Neuron_matrix_weights[w][h][position1] );
                line.append( "," );
                line.append( Neuron_matrix_weights[w][h][position2] );
                line.append( "," );
                line.append( Neuron_matrix_hits[w][h] );
                line.append( "," );
                line.append( w );
                line.append( "," );
                line.append( h );

                lines[neuronCount + 1] = line.toString();
                neuronCount++;
            }
        }
        return lines;
    }
    //</editor-fold>
}
