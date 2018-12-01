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
public class KohonenNeuron 
{
    // .Weight
    double[] Weights;
    
    double LearningFactor;
    int Delay;
    String Name;
    // .Neighbours The connected neurons
    ArrayList<KohonenNeuron> Neighbours;
    
    int Hits;
    
    public KohonenNeuron( int size, String Name )
    {
        this.Weights = new double[size];
        this.Name = Name;
        
        for(int i = 0; i < size; i++)
            this.Weights[i] = Math.random();
        
        this.Neighbours = new ArrayList<>();
        this.Weight_gradients_last = new double[size];
        Arrays.fill(Weight_gradients_last, 0);
        
        this.Hits = 0;
    }
    
    public KohonenNeuron(int size, double[] weights, String Name )
    {
        this.Neighbours = new ArrayList<>();
        
        this.Weights = Arrays.copyOf(weights, weights.length);
        
        double ws = 0;
        for(int i = 0; i < weights.length; i++)
             ws += weights[i];
        
        this.Name = Name + ":" + ((int)(ws * 1000));
        
        this.Weight_gradients_last = new double[size];
        Arrays.fill(Weight_gradients_last, 0);
        
        this.Hits = 0;
    }
    
    public void ConnectTo(KohonenNeuron neighbour)
    {
        Neighbours.add(neighbour);
        neighbour.HiNeighbour(this);
    }
    
    protected void HiNeighbour(KohonenNeuron neighbour)
    {
        Neighbours.add(neighbour);
    }
    
    public void Disconnect(KohonenNeuron neighbour)
    {
        Neighbours.remove(neighbour);
    }
    
    public double[] GetWeights()
    {
        return this.Weights;
    }
    
    public double MeasureDistance( Double[] x )
    {
        double distance = 0d;
        double neighbourCount = (double) this.Neighbours.size();
        
        for(int i = 0; i < Weights.length; i++)
        {
            if( !x[i].isNaN() )
                distance += Math.pow( Weights[i] - x[i], 2d );
            /*
            else
            {
                double n_dist = 0d;
                for (Iterator<KohonenNeuron> it = this.Neighbours.iterator(); it.hasNext();) 
                {
                    n_dist += Math.pow( Weights[i] - it.next().Weights[i], 2d );
                }
                distance += (n_dist /  neighbourCount);
            }
            */
        }

        return Math.pow(distance, 0.5d);
    }
    
    public double MeasureDistance( KohonenNeuron n )
    {
        double distance = 0d;
        for(int i = 0; i < Weights.length; i++)
        {
            distance += Math.pow( Weights[i] - n.Weights[i], 2d );
        }
        return Math.pow(distance, 0.5d);
    }
    
    protected double[] Weight_gradients_last;
    public void Update(double learningRate_t, double neighbourhood_dist, Double[] x )
    {
        // m(i + 1) = m(i) + a(t) * Neighbourhood( c, i ) * ( x - m(i))
        for(int i = 0; i < Weights.length; i++)
        {
            
            if( !x[i].isNaN() )
            {
                Weight_gradients_last[i] = learningRate_t * neighbourhood_dist * ( x[i] - Weights[i] );
                Weights[i] = Weights[i] + Weight_gradients_last[i];
            }
            else // If unknown, do as last time
            {
                Weights[i] = Weights[i] + Weight_gradients_last[i];
                Arrays.fill(Weight_gradients_last, 0d);
            }
        }
    }
    
}
