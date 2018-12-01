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

/**
 * 
 */
public class Axon 
{
    public static final boolean DEV_MODE = false;
    
    // .Weight
    double Weight;
    // .Stimulus
    double Activity;
    
    double Descent;
    
    Neuron Input;
    Neuron Output;
    
    public Axon(Neuron input, Neuron output, double weight)
    {
        this.Input = input;
        this.Output = output;
        this.Weight = weight;
    }
    
    public void Stimulate(double val)
    {
        Activity = val * Weight;
    }
    
    public double GetActivity()
    {
        return Activity;
    }
    
    public double GetWeight()
    {
        return Weight;
    }
    
    public Neuron GetOutput()
    {
        return Output;
    }
    
    public void PropagateError()
    {
        Descent = Input.GetActivity() * Output.GetDescent();
    }
    
    public void UpdateWeights(double learningRate)
    {
        this.Weight += Descent * learningRate;
    }
    
    public double GetDescent()
    {
        return Descent;
    }
    
}
