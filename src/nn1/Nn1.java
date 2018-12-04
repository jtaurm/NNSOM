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

        TestNeuralNetwork( tData, predictors, columnIOMapping);
                
        /*
        // Seed function
        
        int seed = 538;
        for(int s = 0; s < seed; s++)
        {
            for(int r = s; r < rowCount; r += seed)
            {
                System.out.print(r + " ");
            }
            System.out.println();
        }
        */
    }
    
    public void TestNeuralNetwork( Table tData, ArrayList<Integer> response_var_idx, int[] columnIOMapping  )
    {
        String[] lines;
        String resultsFilename;
        
        int inputSize = columnIOMapping.length - response_var_idx.size();
        int outputSize = response_var_idx.size();
        
        double[][] training_data = tData.GetNormalizedDataSetAsArray_rc();
        
        NeuralNetwork_feedforward_leakyReLu nn_test = new NeuralNetwork_feedforward_leakyReLu();
        
        nn_test.Initialize(inputSize, outputSize, 16, 4);
        
        System.out.println("Created NN w " + "many" + " neurons.");
        System.out.println("Training NN");
        
        double rate_start = 0.1;
        double rate_end = 0.0005;
        int rounds = 200;
        int rates = (int) Math.floor( Math.log(rate_start) - Math.log(rate_end) ) + 1;
        
        boolean monte_carlo = true;
        boolean batch_mode = false;
        
        lines = new String[rates*rounds+1];
        lines[0] = "p0,p1,lr,dead";
        int line_no = 1;
        
        for(double learning_rate = rate_start; learning_rate > rate_end; learning_rate *= 1 / Math.E )
        {
            double[] err = new double[line_no];
            int dead = 0;
            
            for(int i = 0; i < rounds; i++)
            {
                err = nn_test.Train(training_data, columnIOMapping, learning_rate, monte_carlo, batch_mode);
                
                dead = nn_test.Count0Neurons();
                lines[line_no] = err[0] + "," + err[1] + "," + learning_rate + "," + dead;
                
                line_no++;
            }
            System.out.println(line_no + "\tlr: " + learning_rate + "\tError: " + err[0] + "\tAct0: " + dead );
        }
        
        // Write training session to file
        resultsFilename = "C:\\Docs\\Statfun\\poe.ninja\\nn.train.err.csv";
        CsvReader.WriteLines(resultsFilename, lines);
                 
        // Predict
        double[][] predictions = nn_test.Predict(training_data, columnIOMapping);
        
        for(int o = 0; o < outputSize; o++)
        {
            Column_Number new_col = new Column_Number("" + tData.GetColumnNameByPosition( response_var_idx.get(o) ) + ".predict" );
            ArrayList<Double> predictions_asCol = new ArrayList<>();
            
            for(int r = 0; r < training_data.length; r++)
                predictions_asCol.add( tData.DenormalizeColumnValue( response_var_idx.get(o), predictions[r][o]) );
            
            new_col.SetData( predictions_asCol );
            
            tData.AddColumn( new_col );
        }
        
        // Write predictions to file
        lines = tData.ToCSVLines(null);
        resultsFilename = "C:\\Docs\\Statfun\\poe.ninja\\currency.piv.all.res.csv";
        CsvReader.WriteLines(resultsFilename, lines);
        
    }
    
    public void TestNeuralNetwork_oldway( Table tData, ArrayList<Integer> response_var_idx, int[] columnIOMapping  )
    {
        String[] lines;
        String resultsFilename;
        
        int inputSize = columnIOMapping.length - response_var_idx.size();
        int outputSize = response_var_idx.size();
        int rowCount = tData.GetRowCount();
        
        double[][] training_data = tData.GetNormalizedDataSetAsArray_rc();
        
        NeuralNetwork_feedforward_leakyReLu nn_test = new NeuralNetwork_feedforward_leakyReLu();
        NeuralNetwork nn_old = new NeuralNetwork();
        
        //nn_test.Initialize(inputSize, outputSize, 16, 4);
        nn_test.Initialize(inputSize, outputSize, 16, 4);
        nn_old.SetupSquare(inputSize, outputSize, 4, 16);
        
        NeuralNetworkTrainer nt_old = new NeuralNetworkTrainer(nn_old);
        nt_old.SetTrainingData(tData, columnIOMapping);
        
        System.out.println("Created NN w " + "many" + " neurons.");
        
        System.out.println("Training NN");
        
        double rate_start = 0.1;
        double rate_end = 0.0005;
        int rounds = 200;
        int rates = (int) Math.floor( Math.log(rate_start) - Math.log(rate_end) ) + 1;
        
        boolean monte_carlo = true;
        boolean batch_mode = false;
        
        lines = new String[rates*rounds+1];
        lines[0] = "p0,p1,lr,dead,p0_old";
        int line_no = 1;
        
        for(double learning_rate = rate_start; learning_rate > rate_end; learning_rate *= 1 / Math.E )
        {
            double[] err = new double[line_no];
            double err_old = 0d;
            int dead = 0;
            
            for(int i = 0; i < rounds; i++)
            {
                err = nn_test.Train(training_data, columnIOMapping, learning_rate, monte_carlo, batch_mode);
                err_old = nt_old.Train(1, learning_rate);
                
                dead = nn_test.Count0Neurons();
                lines[line_no] = err[0] + "," + err[1] + "," + learning_rate + "," + dead + "," + err_old;
                
                line_no++;
                //System.out.println(line_no + "\t" + err[0] + "\t" + learning_rate + "\t" + dead);
            }
            System.out.println(line_no + "\tlr: " + learning_rate + "\tError: " + err[0] + "\tdead: " + dead + "\tError_old:" + err_old);
        }
        
        /*
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
        */
        
        resultsFilename = "C:\\Docs\\Statfun\\poe.ninja\\nn.train.err.csv";
        CsvReader.WriteLines(resultsFilename, lines);
        
        
        //nn_test.Train(training_data, columnIOMapping, 2, 0.01d);
         
        // Predict
        double[][] predictions = nn_test.Predict(training_data, columnIOMapping);
        ArrayList<ArrayList<Double>> pred_old = nt_old.Predict();
        
        for(int o = 0; o < outputSize; o++)
        {
            Column_Number new_col = new Column_Number("" + tData.GetColumnNameByPosition( response_var_idx.get(o) ) + ".predict" );
            ArrayList<Double> predictions_asCol = new ArrayList<>();
            
            for(int r = 0; r < training_data.length; r++)
                predictions_asCol.add( tData.DenormalizeColumnValue( response_var_idx.get(o), predictions[r][o]) );
            
            new_col.SetData( predictions_asCol );
            
            tData.AddColumn( new_col );
        }
        
        
        for(int o = 0; o < outputSize; o++)
        {
            Column_Number new_col = new Column_Number("" + tData.GetColumnNameByPosition( response_var_idx.get(o) ) + ".predict_old" );
            new_col.SetData( pred_old.get(o) );
            tData.AddColumn( new_col );
        }
        
        // 
        lines = tData.ToCSVLines(null);
        resultsFilename = "C:\\Docs\\Statfun\\poe.ninja\\currency.piv.all.res.csv";
        CsvReader.WriteLines(resultsFilename, lines);
        
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
        
        
        
        
    }
    
    public void TestKohonenMap(Table tData )
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

