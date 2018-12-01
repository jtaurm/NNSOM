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
package nn1;

import nn1.neuralnetwork.*;

import java.util.*;
import java.time.*;
import java.time.format.*;
import java.text.*;

/**
 *
 */
public class Nn1 
{
    public static final boolean DEV_MODE = true;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) 
    {
        Nn1 runner = new Nn1();
        runner.Run();
    }
    
    public void Run()
    {
        // Read data
        String filename = "C:\\Docs\\Statfun\\poe.ninja\\currency.piv.all.csv";
        
        if(Nn1.DEV_MODE) System.out.println("Reading data " + filename);
        
        Table tData = CsvReader.ReadAllAsDouble(filename, "[,;]", true);
        
        int tableWidth = tData.GetTableWidth();
        int rowCount = tData.GetRowCount();
        
        if(Nn1.DEV_MODE) System.out.println("Table is " + tableWidth + " variables wide with "+ rowCount +" data rows.");;
        
        String[] lines;
        String resultsFilename;
        
        lines = tData.ToCSVLines_norm_data( null );
        resultsFilename = "C:\\Docs\\Statfun\\poe.ninja\\SOM.dim.norm.csv";
        CsvReader.WriteLines(resultsFilename, lines);
        
        // Configure data
        
        int IdxExOrb = tData.GetPositionByColumnName("Exalted Orb");
        int IdxExShd = tData.GetPositionByColumnName("Exalted Shard");
        
        ArrayList<Integer> predictors = new ArrayList<>();
        predictors.add(IdxExOrb);
        predictors.add(IdxExShd);
        
        int[] columnIOMapping = new int[tData.GetTableWidth()];
        for(int i = 0; i < tData.GetTableWidth(); i++)
        {
            if(predictors.contains(i))
                columnIOMapping[i] = -1;
            else
                columnIOMapping[i] = 1;
        }
        
        if(Nn1.DEV_MODE) System.out.println("exes at " + IdxExOrb + " and " + IdxExShd );
        
        // Test PCA
        
        /*
        if(Nn1.DEV_MODE) System.out.println("Generating PC1 & 2");
        
        ArrayList<double[]> pc_list = tData.GetPCs(0.00001, 3);
        for( double[] pc : pc_list)
        {
            System.out.print("pc\t");
            for(int i = 0; i < pc.length; i++)
                System.out.print( pc[i] + ",");
            System.out.println();
        }
        */
        
        // Test SOM
        
        TestKohonenMap( tData );
        
        // Test NN

        //TestNeuralNetwork( tData, predictors, columnIOMapping);
    }
    
    public void TestNeuralNetwork( Table tData, ArrayList<Integer> predictors, int[] columnIOMapping  )
    {
        int inputSize = tData.GetTableWidth() - predictors.size();
        int outputSize = predictors.size();
        
        NeuralNetwork nn_test = new NeuralNetwork( );
        nn_test.SetupSquare( inputSize, outputSize, 4, 16 );
        //nn_test.SetupLogSize(inputSize, outputSize, Math.E );
        
        System.out.println("Created NN w " + nn_test.GetFeedOrder().size() + " neurons.");
        
        NeuralNetworkTrainer nn_trainer = new NeuralNetworkTrainer( nn_test );
        nn_trainer.SetTrainingData( tData, columnIOMapping );
        
        System.out.println("Training NN");
        
        // Train
        nn_trainer.Train_k(100, 0.1, 10);
        nn_trainer.Train_k(100, 0.05, 10);
        nn_trainer.Train_k(100, 0.01, 10);
        nn_trainer.Train_k(100, 0.006, 10);
        nn_trainer.Train_k(100, 0.003, 10);
        nn_trainer.Train_k(100, 0.002, 10);
        nn_trainer.Train_k(100, 0.001, 10);
        nn_trainer.Train_k(100, 0.0009, 10);
        nn_trainer.Train_k(100, 0.0008, 10);
        nn_trainer.Train_k(100, 0.0005, 10);
        
        
        /*
        // Map input strength
        ArrayList<double[]> inputStrength = nn_trainer.MapInputStrength();
        
        NumberFormat format3d = new DecimalFormat("#0.000");     

        for(int c = 0; c < inputStrength.size(); c++)
        {
            for(int o = 0; o < outputSize; o++)
            {
                System.out.println(
                        "Strength of input: " + tData.GetColumnNameByPosition(c) + 
                                " is\t" + format3d.format(inputStrength.get(c)[o]) + " for output: " + 
                                tData.GetColumnNameByPosition( predictors.get(o) ) 
                );
            }
        }
        */
        
        // Predict
        ArrayList<ArrayList<Double>> columnPredictions = nn_trainer.Predict();
        
        for(int i = 0; i < columnPredictions.size(); i++)
        {
            Column_Number new_col = new Column_Number("" + tData.GetColumnNameByPosition( predictors.get(i) ) + ".predict" );
            new_col.SetData( columnPredictions.get(i) );
            
            tData.AddColumn( new_col );
        }
        
        // 
        String[] lines = tData.ToCSVLines(null);
        
        String resultsFilename = "C:\\Docs\\Statfun\\poe.ninja\\currency.piv.all.res.csv";
        CsvReader.WriteLines(resultsFilename, lines);
    }
    
    public void TestKohonenMap( Table tData )
    {
        double learning_rate = 0.05;
        double mse_delta_log_threshold = 3d;
        int coop_radius = 3;
        double rowCount_rate = 1.0;
        
        int duration_avg = 4;
        
        TestKohonenMap( tData, learning_rate, coop_radius, rowCount_rate, mse_delta_log_threshold, duration_avg);
    }
    
    public void TestKohonenMap(Table tData, double learning_rate, int coop_radius, double rowCount_rate, double mse_delta_log_threshold, int duration_avg )
    {
        SelfOrganizingMap_arr_hc_toroid missingInputPredictor = new SelfOrganizingMap_arr_hc_toroid( tData );
        missingInputPredictor.Initialize();
        
        String[] lines;
        String resultsFilename;
        
        lines = missingInputPredictor.PrintDimension(32, 17);
        resultsFilename = "C:\\Docs\\Statfun\\poe.ninja\\SOM.init.csv";
        CsvReader.WriteLines(resultsFilename, lines);
        
        System.out.println("Training SOM");
        Double mse = missingInputPredictor.Train_Auto(true);
        System.out.println("Trained SOM - final error rate " + mse);
        
        missingInputPredictor.ClearUnused();
        System.out.println("Cleaned SOM " + missingInputPredictor.GetNeuronCount() + " neurons left.");
        
        missingInputPredictor.FillMissingValues();
        
        lines = missingInputPredictor.PrintDimension(32, 17);
        resultsFilename = "C:\\Docs\\Statfun\\poe.ninja\\SOM.dim2.t.csv";
        CsvReader.WriteLines(resultsFilename, lines);
    }
    
}

