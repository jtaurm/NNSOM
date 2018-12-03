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

import java.util.*;

/**
 *
 */
public class NeuralNetwork 
{
    public static final boolean DEV_MODE = false;
    
    /**
     * Neurons
     */
    private final ArrayList<Neuron> LayerInput;
    private final ArrayList<ArrayList<Neuron>> LayerListHidden;
    private final ArrayList<Neuron> LayerOutput;
    
    public NeuralNetwork()
    {
        LayerInput = new ArrayList<>();
        LayerListHidden = new ArrayList<>();
        LayerOutput = new ArrayList<>();
        FeedOrder = new ArrayList<>();
        FeedOrderRev = new ArrayList<>();
        FeedOrderNoInput = new ArrayList<>();
        FeedOrderNoInputRev = new ArrayList<>();
    }
       
    /**
     * Classic grid based network
     * @param inputSize
     * @param outputSize
     * @param depth Number of hidden layers
     * @param width Number of neurons per layer
     */
    public void SetupSquare( int inputSize, int outputSize, int depth, int width )
    {
        // Create input neurons
        for(int i = 0; i < inputSize; i++)
        {
            Neuron newInputNeuron = new Neuron( null, "i" + i, Neuron.ActivationFunctionType.LeakyReLu );
            LayerInput.add( newInputNeuron );
            FeedOrder.add( newInputNeuron );
        }
        
        // Create hidden layers
        for(int i = 0; i < depth; i++)
        {
            ArrayList<Neuron> LayerHidden = new ArrayList<>();
            LayerListHidden.add( LayerHidden );
            
            for(int j = 0; j < width; j++)
            {
                Neuron newNeuron = new Neuron( null, "h" + i + "," + j, Neuron.ActivationFunctionType.LeakyReLu );
                LayerHidden.add(newNeuron);
                FeedOrder.add( newNeuron );
                FeedOrderNoInput.add( newNeuron );
                
                if(i == 0)
                {
                    // First hidden layer
                    for(int k = 0; k < inputSize; k++)
                    {
                        // Get input layer neuron
                        Neuron inputNeuron = LayerInput.get(k);
                        newNeuron.ConnectInput( inputNeuron, (Math.random() + 0.5) / inputSize ); 
                        
                    }
                    
                }
                else
                {
                    // Deep hidden layer
                    ArrayList<Neuron> LastLayer = LayerListHidden.get(i - 1);
                    
                    for(int k = 0; k < width; k++)
                    {
                        // Get last hidden layer neuron
                        Neuron upNeuron = LastLayer.get(k);
                        
                        // Connect
                        newNeuron.ConnectInput( upNeuron, (Math.random() + 0.5) / width ); 
                        
                    }
                    
                }
                                
            }
        }
        
        // Create output layer
        ArrayList<Neuron> LastLayer = LayerListHidden.get(depth - 1);
        
        for(int i = 0; i < outputSize; i++)
        {
            Neuron newNeuron = new Neuron( null, "o" + i, Neuron.ActivationFunctionType.None );
            LayerOutput.add(newNeuron);
            FeedOrder.add( newNeuron );
            FeedOrderNoInput.add( newNeuron );
            
            for(int k = 0; k < width; k++)
            {
                 // Get last hidden layer neuron
                Neuron upNeuron = LastLayer.get(k);

                // Connect
                newNeuron.ConnectInput( upNeuron, (Math.random() + 0.5) / width );
            }
        }
        
        FeedOrderRev.addAll( FeedOrder );
        Collections.reverse(FeedOrderRev );
        
        FeedOrderNoInputRev.addAll( FeedOrderNoInput );
        Collections.reverse(FeedOrderNoInputRev );
        
        
    }
    
    
    public void SetupLogSize( int inputSize, int outputSize )
    {
        SetupLogSize(inputSize, outputSize, Math.E );
    }
    
    /**
     * Log-based layer size. Each subsequent layer is the size of the previous layer / scaling.
     * @param inputSize
     * @param outputSize 
     * @param scaling
     */
    public void SetupLogSize( int inputSize, int outputSize, double scaling )
    {
        // Create input neurons
        for(int i = 0; i < inputSize; i++)
        {
            Neuron newInputNeuron = new Neuron( null, "i" + i, Neuron.ActivationFunctionType.LeakyReLu );
            LayerInput.add( newInputNeuron );
            FeedOrder.add( newInputNeuron );
        }
        
        // Create hidden layers
        int width = (int) Math.ceil( (double) inputSize * (1/scaling) );
        int i = 0; // hidden layer no
        
        while(width > outputSize)
        {
            ArrayList<Neuron> LayerHidden = new ArrayList<>();
            LayerListHidden.add( LayerHidden );
            
            for(int j = 0; j < width; j++)
            {
                Neuron newNeuron = new Neuron( null, "h" + i + "," + j, Neuron.ActivationFunctionType.LeakyReLu );
                LayerHidden.add(newNeuron);
                FeedOrder.add( newNeuron );
                FeedOrderNoInput.add( newNeuron );
                
                if(i == 0)
                {
                    // First hidden layer
                    for(int k = 0; k < inputSize; k++)
                    {
                        // Get input layer neuron
                        Neuron inputNeuron = LayerInput.get(k);
                        newNeuron.ConnectInput( inputNeuron, (Math.random() + 0.5) / inputSize ); 
                        
                    }
                    
                }
                else
                {
                    // Deep hidden layer
                    ArrayList<Neuron> LastLayer = LayerListHidden.get(i - 1);
                    
                    for(int k = 0; k < LastLayer.size(); k++)
                    {
                        // Get last hidden layer neuron
                        Neuron upNeuron = LastLayer.get(k);
                        
                        // Connect
                        newNeuron.ConnectInput( upNeuron, (Math.random() + 0.5) / width ); 
                        
                    }
                    
                }
                                
            }
            
            width = (int) Math.ceil( (double)width * (1/scaling) );
            i++;
        }
        
        int depth = i;
        
        // Create output layer
        ArrayList<Neuron> LastLayer = LayerListHidden.get(depth - 1);
        
        for(i = 0; i < outputSize; i++)
        {
            Neuron newNeuron = new Neuron( null, "o" + i, Neuron.ActivationFunctionType.None );
            LayerOutput.add(newNeuron);
            FeedOrder.add( newNeuron );
            FeedOrderNoInput.add( newNeuron );
            
            for(int k = 0; k < LastLayer.size(); k++)
            {
                 // Get last hidden layer neuron
                Neuron upNeuron = LastLayer.get(k);

                // Connect
                newNeuron.ConnectInput( upNeuron, (Math.random() + 0.5) / width );
            }
        }
        
        FeedOrderRev.addAll( FeedOrder );
        Collections.reverse(FeedOrderRev );
        
        FeedOrderNoInputRev.addAll( FeedOrderNoInput );
        Collections.reverse(FeedOrderNoInputRev );
        
        
    }
    
    public ArrayList<Neuron> GetOutputNeurons()
    {
        return LayerOutput;
    }
    
    final ArrayList<Neuron> FeedOrder;
    final ArrayList<Neuron> FeedOrderRev;
    
    final ArrayList<Neuron> FeedOrderNoInput;
    final ArrayList<Neuron> FeedOrderNoInputRev;
    
    public ArrayList<Neuron> GetFeedOrder()
    {
        return FeedOrder;
    }
    
    public ArrayList<Neuron> FeedInputs( Double[] inputs )
    {
        // Stimulate input neurons
        for(int i = 0; i < Math.min( LayerInput.size(), inputs.length); i++)
        {
            LayerInput.get(i).Stimulate( inputs[i] );
        }
        
        // Run feed forward
        for (Iterator<Neuron> it = FeedOrderNoInput.iterator(); it.hasNext();) {
            Neuron n = it.next();
            
            n.FeedForward();
        }
                
        // Return output layer for user to read
        return LayerOutput;
    }
    
    public void PropagateError( Double[] errors, double learningRate )
    {
        if(nn1.Nn1.DEV_MODE && DEV_MODE) System.out.println("Setting cost");
        
        // Stimulate input neurons
        for(int i = 0; i < Math.min( LayerOutput.size(), errors.length); i++)
        {
            LayerOutput.get(i).SetCost( 2 * errors[i] );
        }
        
        if(nn1.Nn1.DEV_MODE && DEV_MODE) System.out.println("Back propagate cost - " + FeedOrderRev.size() + " Neurons."  );
        
        // Run back propagate
        for (Iterator<Neuron> it = FeedOrderRev.iterator(); it.hasNext();) 
        {
            Neuron n = it.next();
            
            n.PropagateError();
        }
        
        if(nn1.Nn1.DEV_MODE && DEV_MODE) System.out.println("Update weights");
        
        // Update weights and biases
        for (Iterator<Neuron> it = FeedOrderRev.iterator(); it.hasNext();) 
        {
            Neuron n = it.next();
            
            n.UpdateWeights(learningRate);
        }
        
    }
    
}
