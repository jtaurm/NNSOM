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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

/**
 *
 */
public class Column_Number extends Column
{
    String Name;
    
    public Column_Number()
    {
        this.Name = "Validator";
    }
    
    public Column_Number( String name )
    {
        this.Name = name;
    }
    
    public Column_Number( String name, ArrayList<String> data )
    {
        this.Name = name;
        
        AddData(data);
        Compile();
    }
    
    public static final Column Validator = new Column_Number();
    
    /**
     *
     * @param data
     * @return
     */
    @Override
    public boolean ValidateType(ArrayList<String> data )
    {
        for(int i = 0; i < data.size(); i++)
        {
            String value = data.get(i);
            
            if( value.equals("NA") )
                    continue;
                
                if( value.equals("NaN") )
                    continue;
                
                if( value.equals("null") )
                    continue;
                
                if( value.equals("NULL") )
                    continue;
                
                if( value.equals("") )
                    continue;
                
            try
            {
                Double.parseDouble(value);
            }
            catch(NumberFormatException e)
            {
                return false;
            }
        }
        
        return true;
    }
    
    ArrayList<Double> DataAsDouble;
    
    /**
     *
     * @param data
     */
    @Override
    public final void AddData( ArrayList<String> data )
    {
        DataAsDouble = new ArrayList<>();
        
        for(int i = 0; i < data.size(); i++)
        {
            String value = data.get(i);
            
            if( 
                    value.equals("NA") ||
                    value.equals("NaN") ||
                    value.equals("null") ||
                    value.equals("NULL") ||
                    value.equals("")
            )
            {
                DataAsDouble.add( Double.NaN );
                continue;
            }

            DataAsDouble.add( Double.parseDouble(value) );
        }
    }
    
    public final void SetData( ArrayList<Double> data )
    {
        DataAsDouble = new ArrayList<>();
        
        for(int i = 0; i < data.size(); i++)
        {
            DataAsDouble.add( data.get(i) );
        }
        
        Compile();
    }
    
    @Override
    public String GetName()
    {
        return Name;
    }
    
    @Override
    public int GetWidth()
    {
        return 1;
    }
    
    @Override
    public int GetRowCount()
    {
        return DataAsDouble.size();
    }
    
    @Override
    public Table.ColumnDataType GetType()
    {
        return Table.ColumnDataType.Numeric;
    }
    
    ArrayList<Double> DataAsDoubleNorm;
    boolean needs_recompile = false;
    
    Double Stat_min;
    Double Stat_max;
    Double Stat_norm_avg;
    Double Stat_norm_var;
    int Stat_nonNa_cnt;
    
    int[] SortIndex;
    
    @Override
    public final void Compile()
    {
        DataAsDoubleNorm = new ArrayList<>();
        
        Stat_min = Double.POSITIVE_INFINITY;
        Stat_max = Double.NEGATIVE_INFINITY;
        
        for(int i = 0; i < DataAsDouble.size(); i++)
        {
            Double value = DataAsDouble.get(i);
            
            if( !value.isNaN() )
            {
                // Min, Max
                if( value > Stat_max )
                    Stat_max = value;

                if( value < Stat_min )
                    Stat_min = value;
            }
        }
        
        // Calc average
        Stat_norm_avg = 0d;
        Stat_nonNa_cnt = 0;
        
        for(int i = 0; i < DataAsDouble.size(); i++)
        {
            Double value = DataAsDouble.get(i);
            
            if( !value.isNaN() )
            {
                Double value_norm = (value - Stat_min) / (Stat_max - Stat_min);

                // Mean
                Stat_norm_avg += value_norm;
                Stat_nonNa_cnt++;
                
                // Normalize
                DataAsDoubleNorm.add(value_norm);
            }
            else
            {
                DataAsDoubleNorm.add(value);
            }
        }
        Stat_norm_avg = Stat_norm_avg / Stat_nonNa_cnt;
        
        // Var
        Stat_norm_var = 0d;
        for(int i = 0; i < DataAsDoubleNorm.size(); i++)
        {
            Double value = DataAsDoubleNorm.get(i);
            if( !value.isNaN() )
            {
                Stat_norm_var += Math.pow(Stat_norm_avg - value, 2d);
            }
        }
        Stat_norm_var = Stat_norm_var / Stat_nonNa_cnt;
        
        // Build sort index
        SortIndex = new int[DataAsDouble.size()];
        Integer[] SortIndexObj = new Integer[DataAsDouble.size()];
        for(int i = 0; i < DataAsDouble.size(); i++)
            SortIndexObj[i] = i;
        
        Arrays.sort( SortIndexObj, new Comparator<Integer>() {
            public int compare(Integer o1, Integer o2) {
                return DataAsDouble.get(o1).compareTo( DataAsDouble.get(o2) );
            }
        });
        
        for(int i = 0; i < DataAsDouble.size(); i++)
            SortIndex[i] = SortIndexObj[i];
        
        // Set flags
        
        needs_recompile = false;
    }
        
    @Override
    public void SetValue_Numeric(int row, int position, Double value_new )
    {
        // Min/Max change requires renormalized column
        if( value_new > Stat_max)
        {
            Stat_max = value_new;
            needs_recompile = true;
            return;
        }
        
        if( value_new < Stat_min)
        {
            Stat_min = value_new;
            needs_recompile = true;
            return;
        }
        
        Double value_old = DataAsDouble.get(row);
        Double value_old_norm = DataAsDoubleNorm.get(row);
        
        // old = new
        if( value_old.equals(value_new))
            return;
        
        
        if( value_old.isNaN() && !value_new.isNaN() )
        {   // old = NaN, new = value
            double value_new_norm = NormalizeValue( value_new );
            
            DataAsDouble.set(row, value_new);
            DataAsDoubleNorm.set(row, value_new_norm );
            
            Stat_norm_avg = ((Stat_norm_avg * Stat_nonNa_cnt) + value_new_norm) / (Stat_nonNa_cnt + 1);
            Stat_norm_var = ((Stat_norm_var * Stat_nonNa_cnt) + Math.pow(Stat_norm_avg - value_new_norm, 2d)) / (Stat_nonNa_cnt + 1);
            
            Stat_nonNa_cnt++;
        }
        else
        if( !value_old.isNaN() && value_new.isNaN() )
        {   // old = value, new = NaN
            DataAsDouble.set(row, Double.NaN );
            DataAsDoubleNorm.set(row, Double.NaN );
            
            Stat_norm_avg = ((Stat_norm_avg * Stat_nonNa_cnt) - value_old_norm) / (Stat_nonNa_cnt - 1);
            Stat_norm_var = ((Stat_norm_var * Stat_nonNa_cnt) - Math.pow(Stat_norm_avg - value_old_norm, 2d)) / (Stat_nonNa_cnt - 1);
            Stat_nonNa_cnt--;
        }
        else
        {   // old = new
            double value_new_norm = NormalizeValue( value_new );
            
            DataAsDouble.set(row, value_new);
            DataAsDoubleNorm.set(row, value_new_norm );
            
            Stat_norm_avg = ((Stat_norm_avg * Stat_nonNa_cnt) + (value_new_norm - value_old_norm) ) / (Stat_nonNa_cnt);
            Stat_norm_var = ((Stat_norm_var * Stat_nonNa_cnt) + Math.pow(Stat_norm_avg - (value_new_norm - value_old_norm), 2d)) / (Stat_nonNa_cnt);
        }
        
    }
    
    @Override
    public Double GetValue_Normalized(int row, int position)
    {
        if(needs_recompile)
            Compile();
        
        return DataAsDoubleNorm.get(row);
    }
    
    @Override
    Double[] GetValues_Normalized( int position )
    {
        int row_count = GetRowCount();
        Double[] arr = new Double[row_count];
        return DataAsDoubleNorm.toArray(arr);
    }
    
    @Override
    public Double GetValue_Numeric(int row, int position)
    {
        return DataAsDouble.get(row);
    }
    
    @Override
    public String GetValue_String(int row)
    {
        return DataAsDouble.get(row).toString();
    }
    
    @Override
    public Double GetValue_Min()
    {
        return Stat_min;
    }
    
    @Override
    public Double GetValue_Max()
    {
        return Stat_max;
    }
    
    @Override
    public Double GetValue_Avg(int position)
    {
        return DenormalizeValue(Stat_norm_avg);
    }
    
    @Override
    public Double GetValue_Var(int position)
    {
        // Does var scale???
        return DenormalizeValue(Stat_norm_var);
    }
    
    @Override 
    public int GetRowIndex_NormalizedSorted(int order, int position)
    {
        return SortIndex[order];
    }
    
    @Override
    public int GetRowIndex_Median(int position)
    {
        return (int) Math.floor( (double) DataAsDouble.size() * 0.5 );
    }
    
    @Override
    public double DenormalizeValue( double value)
    {
        return (value * ( Stat_max - Stat_min ) ) + Stat_min;
    }
    
    @Override
    public double NormalizeValue( double value)
    {
        return (value - Stat_min) / ( Stat_max - Stat_min );
    }
    
}
