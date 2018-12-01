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
import java.util.HashSet;
import java.util.Iterator;

/**
 *
 */
public class Column_Levels extends Column
{
    String Name;
    
    public Column_Levels()
    {
        this.Name = "Validator";
    }
    
    public Column_Levels(String name)
    {
        this.Name = name;
    }
    
    public Column_Levels( String name, ArrayList<String> data )
    {
        this.Name = name;
        
        this.AddData(data);
        this.Compile();
    }
    
    public static final Column Validator = new Column_Levels();
    
    /**
     *
     * @param data
     * @return
     */
    @Override
    public boolean ValidateType(ArrayList<String> data )
    {
        return true;
    }
    
    ArrayList<Boolean[]> DataAsBools;
    String[] LevelValues;
    
    /**
     *
     * @param data
     */
    @Override
    public final void AddData( ArrayList<String> data )
    {
        // Convert Levels into Byte array
        HashSet UniqueValuesSet = new HashSet();
        ArrayList<String> UniqueValueList = new ArrayList<>();

        for (Iterator<String> itCol = data.iterator(); itCol.hasNext(); ) 
        {
            String value = itCol.next();
            
            if( UniqueValuesSet.contains(value) )
                continue;
            
            UniqueValuesSet.add(value);
            UniqueValueList.add(value);
        }

        LevelValues = (String[]) UniqueValueList.toArray(new String[UniqueValuesSet.size()]);

        // Copy column levels/factors into bit array
        DataAsBools = new ArrayList<>();

        for( int lineIdx = 0; lineIdx < data.size(); lineIdx++)
        {
            Boolean[] RowAsBools = new Boolean[ LevelValues.length ];
            Arrays.fill(RowAsBools, false);

            DataAsBools.add(RowAsBools);

            for( int i = 0; i < LevelValues.length; i++)
            {
                if(data.get(lineIdx).equals(LevelValues[i]))
                {
                    RowAsBools[i] = true;
                }

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
        return LevelValues.length;
    }
    
    @Override
    public int GetRowCount()
    {
        return DataAsDouble.size();
    }
    
    @Override
    public Table.ColumnDataType GetType()
    {
        return Table.ColumnDataType.Levels;
    }
    
    public String[] GetLevels()
    {
        return LevelValues;
    }
    
    ArrayList<Double[]> DataAsDouble;
    
    Double[] Stat_avg;
    Double[] Stat_var;
    
    @Override
    public final void Compile()
    {
        // Convert to double
        DataAsDouble = new ArrayList<>();
        
        for(int i = 0; i < DataAsBools.size(); i++)
        {
            Double[] row = new Double[ LevelValues.length ];
            
            for(int p = 0; p < LevelValues.length; p++)
            {
                if( DataAsBools.get(i)[p] )
                    row[p] = 1.0d;
                else
                    row[p] = 0.0d;
            }
            
            DataAsDouble.add(row);
            
        }
        
        // Generate stats
                
        // Mean
        Stat_avg = new Double[ LevelValues.length ];
        Arrays.fill(Stat_avg, 0d);
        
        for(int i = 0; i < DataAsBools.size(); i++)
        {
            for(int p = 0; p < LevelValues.length; p++)
            {
                Stat_avg[p] += DataAsDouble.get(i)[p];
            }
        }
        
        for(int p = 0; p < LevelValues.length; p++)
        {
            Stat_avg[p] = Stat_avg[p] / DataAsBools.size();
        }
        
        // Var
        Stat_var = new Double[ LevelValues.length ];
        Arrays.fill(Stat_var, 0d);
        
        for(int i = 0; i < DataAsBools.size(); i++)
        {
            for(int p = 0; p < LevelValues.length; p++)
            {
                Stat_var[p] += Math.pow(Stat_avg[p] - DataAsDouble.get(i)[p], 2d);
            }
        }
        
        for(int p = 0; p < LevelValues.length; p++)
        {
            Stat_var[p] = Stat_var[p] / DataAsBools.size();
        }
    }
    
    @Override
    public void SetValue_Numeric(int row, int position, Double value_new )
    {
        Double value_old = DataAsDouble.get(row)[position];
        Double value_old_norm = value_old;
        
        if( !value_old.equals(value_new))
        {   // old != new
            double value_new_norm = value_new;
            
            DataAsDouble.get(row)[position] = value_new;
            
            Stat_avg[position] = ((Stat_avg[position] * DataAsBools.size()) + (value_new - value_old) ) / (DataAsBools.size());
            Stat_var[position] = ((Stat_var[position] * DataAsBools.size()) + Math.pow(Stat_avg[position] - (value_new - value_old), 2d)) / (DataAsBools.size());
        }
    }
    
    @Override
    public Double GetValue_Normalized(int row, int position)
    {
        return DataAsDouble.get(row)[position];
    }
    
    @Override
    public Double GetValue_Numeric(int row, int position)
    {
        return DataAsDouble.get(row)[position];
    }
    
    @Override
    public String GetValue_String(int row)
    {
        Boolean[] values = DataAsBools.get(row);
        for(int p = 0; p < LevelValues.length; p++)
        {
            if( values[p] )
                return LevelValues[p];
        }
        
        return "";
    }
    
    @Override
    public Double GetValue_Min()
    {
        return 0d;
    }
    
    @Override
    public Double GetValue_Max()
    {
        return 1d;
    }
    
    @Override
    public Double GetValue_Avg(int position)
    {
        return Stat_avg[position];
    }
    
    @Override
    public Double GetValue_Var(int position)
    {
        // Does var scale???
        return Stat_var[position];
    }
    
    @Override
    public double DenormalizeValue( double value)
    {
        return value;
    }
    
    @Override
    public double NormalizeValue( double value)
    {
        return value;
    }
}

