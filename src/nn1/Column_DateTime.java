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

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;

/**
 *
 */
public class Column_DateTime extends Column
{

    String Name;
    
    public Column_DateTime()
    {
        this.Name = "Validator";
    }
    
    public Column_DateTime(String name)
    {
        this.Name = name;
    }
    
    public Column_DateTime( String name, ArrayList<String> data )
    {
        this.Name = name;
        
        AddData(data);
        Compile();
    }
    
    public static final Column Validator = new Column_DateTime();
    
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
            try
            {
                if( value.equals("NULL") )
                    continue;

                if( value.equals("null") )
                    continue;

                if( value.length() == 0 )
                    continue;
                
                if( value.length() > 13 )
                {
                    LocalDateTime.parse(value, DateTimeFormatter.ISO_DATE_TIME);
                }
                else
                {
                    LocalDate.parse(value, DateTimeFormatter.ISO_DATE);
                }
            }
            catch(java.time.format.DateTimeParseException e)
            {
                return false;
            }
        }
        
        return true;
    }
    
    ArrayList<LocalDateTime> DataAsDateTime;
    
    /**
     *
     * @param data
     */
    @Override
    public final void AddData( ArrayList<String> data )
    {
        DataAsDateTime = new ArrayList<>();
        
        for(int i = 0; i < data.size(); i++)
        {
            String value = data.get(i);
            
            if( value.equals("NULL") )
                DataAsDateTime.add( null );
            
            if( value.equals("null") )
                DataAsDateTime.add( null );
            
            if( value.length() == 0 )
                DataAsDateTime.add( null );
            
            if( value.length() > 13 )
            {
                DataAsDateTime.add( LocalDateTime.parse(value, DateTimeFormatter.ISO_DATE_TIME) );
            }
            else
            {
                DataAsDateTime.add( LocalDate.parse(value, DateTimeFormatter.ISO_DATE).atTime(0, 0, 0) );
            }
            
        }
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
        return Table.ColumnDataType.Datetime;
    }
    
    
    ArrayList<Double> DataAsDouble;
    ArrayList<Double> DataAsDoubleNorm;
    boolean needs_recompile = false;
    
    Double Stat_min;
    Double Stat_max;
    Double Stat_norm_avg;
    Double Stat_norm_var;
    int Stat_nonNa_cnt;
    
    @Override
    public final void Compile()
    {
        
        // Convert to double
        DataAsDouble = new ArrayList<>();
        
        Stat_min = Double.POSITIVE_INFINITY;
        Stat_max = Double.NEGATIVE_INFINITY;
        Stat_nonNa_cnt = 0;
        
        for(int i = 0; i < DataAsDateTime.size(); i++)
        {
            LocalDateTime value_dt = DataAsDateTime.get(i);
            Double value;
            
            if(value_dt == null)
                value = Double.NaN;
            else
            {
                value = (double) value_dt.toEpochSecond(ZoneOffset.UTC);
                Stat_nonNa_cnt++;
                
                // Min, Max
                if( value > Stat_max )
                    Stat_max = value;

                if( value < Stat_min )
                    Stat_min = value;
            }
            
            DataAsDouble.add(value);
        }
        
        // Mean
        DataAsDoubleNorm = new ArrayList<>();
        
        Stat_norm_avg = 0d;
        for(int i = 0; i < DataAsDouble.size(); i++)
        {
            Double value = DataAsDouble.get(i);
            
            if(!value.isNaN())
            {
                Double value_norm = (value - Stat_min) / (Stat_max - Stat_min);
                
                DataAsDoubleNorm.add(value_norm);
                Stat_norm_avg += value_norm;
            }
            else
            {
                DataAsDoubleNorm.add(Double.NaN);
            }
            
        }
        Stat_norm_avg = Stat_norm_avg / Stat_nonNa_cnt;
        
        // Var
        Stat_norm_var = 0d;
        for(int i = 0; i < DataAsDoubleNorm.size(); i++)
        {
            Double value_norm = DataAsDoubleNorm.get(i);
            
            if(!value_norm.isNaN())
            {
                Stat_norm_var += Math.pow(Stat_norm_avg - value_norm, 2d);
            }
        }
        Stat_norm_var = Stat_norm_var / Stat_nonNa_cnt;
    }
    
    @Override
    public void SetValue_Numeric(int row, int position, Double value_new )
    {
        // Set DAtetime valu
        //int dec = (int) value;
        //double nan = (value - dec) * 1000000000;
        
        
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
    public Double GetValue_Numeric(int row, int position)
    {
        return DataAsDouble.get(row);
    }
    
    @Override
    public String GetValue_String(int row)
    {
        LocalDateTime value = DataAsDateTime.get(row);
        
        if(value == null)
            return "";
        else
            return DataAsDateTime.get(row).toString();
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
