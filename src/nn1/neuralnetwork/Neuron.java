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
public class Neuron 
{
    public static final boolean DEV_MODE = false;
    
    // .Inputs
    ArrayList<Axon> Inputs;
    
    // .Outputs
    ArrayList<Axon> Outputs;
    
    // .Bias
    double Bias;
    double Activity;
    
    double dZ;
    
    String Name;
    ActivationFunctionType ActivationFunction;
    
    enum ActivationFunctionType
    {
        Sigmoid, ReLu, LeakyReLu, None
    }
         
    
    public Neuron(Double Bias, String name, ActivationFunctionType activationFunc )
    {
        if(Bias == null)
        {
            double r = Math.random();
            Bias = (r > 0.5 ? r - 0.25 : -0.25 - r);
        }
        
        Inputs = new ArrayList<>();
        Outputs = new ArrayList<>();
        this.Bias = Bias;
        this.Name = name;
        this.ActivationFunction = activationFunc;
        
        //System.out.println("Neuron: " + Name);
    }
    
    public void AddOutput(Axon output)
    {
        Outputs.add(output);
    }
    
    // fConnectInput
    public void ConnectInput( Neuron input, double weight )
    {
        Axon connection = new Axon(input, this, weight);
        Inputs.add(connection);
        input.AddOutput(connection);
    }
    
    // fForward
    public ArrayList<Axon> FeedForward()
    {
        // get sum stimulus from inputs
        double stimulus_sum = 0;
        
        for(int i = 0; i < Inputs.size(); i++)
        {
            stimulus_sum += Inputs.get(i).GetActivity();
        }
        
        return Stimulate(stimulus_sum);
    }
    
    public ArrayList<Axon> Stimulate(Double stimulus)
    {
        if (stimulus == null || stimulus.isNaN())
        {
            Activity = 0.0;
        }
        else
        {
            switch(ActivationFunction)
            {
                case LeakyReLu:
                    Activity = (stimulus + Bias > 0 ? stimulus + Bias : (stimulus + Bias) *0.1); // Leaky ReLu
                    break;
                case ReLu:
                    Activity = Math.max(0, stimulus + Bias); // ReLu
                    break;
                case Sigmoid:
                    Activity = 1.0 / (1.0 + Math.pow( Math.E, -(stimulus + Bias) )); // Sigmoid
                case None:
                    Activity = stimulus + Bias;
            }
            
        }
        
        // stimulate output
        for(int i = 0; i < Outputs.size(); i++)
        {
            Outputs.get(i).Stimulate(Activity);
        }
        
        return Outputs;
    }
    
    public void SetCost(Double cost)
    {
        this.Cost = cost;
    }
    
    Double Cost;
    
    public void PropagateError()
    {
        double dAct;
        if( Cost != null && !Cost.isNaN() )
        {
            dAct = Cost; //2*(Activity - Cost);
        }
        else
        {
            dAct = 0;
            for(int i = 0; i < Outputs.size(); i++)
            {
                dAct += Outputs.get(i).GetWeight() * Outputs.get(i).GetOutput().GetDescent();
            }
        }
        
        switch(ActivationFunction)
        {
            case LeakyReLu:
                dZ = (Activity > 0 ? dAct : dAct * 0.1); // Leaky ReLu
                break;
            case ReLu:
                dZ = (Activity > 0 ? dAct : 0); // ReLu
                break;
            case Sigmoid:
                dZ = Activity * (1 - Activity) * dAct; // Sigmoid
            case None:
                dZ = dAct; // None/linear
        }
        
        // propagate to weights
        for(int i = 0; i < Inputs.size(); i++)
        {
            Inputs.get(i).PropagateError();
        }
        
    }
    
    public void UpdateWeights(double learningRate)
    {
        Bias += dZ * learningRate;
        
        for(int i = 0; i < Inputs.size(); i++)
        {
            Inputs.get(i).UpdateWeights(learningRate);
        }
    }
        
    /**
     * 
     * @return Activity of Neuron
     */
    public double GetActivity()
    {
        return Activity;
    }
    
    /**
     * 
     * @return dZ - the descent
     */
    public double GetDescent()
    {
        return dZ;
    }
    
    public ArrayList<Axon> GetOutputs()
    {
        return Outputs;
    }
    
}
