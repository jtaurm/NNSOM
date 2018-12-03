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
import java.util.*;

/**
 *
 */
public class Table 
{
    public static final boolean DEV_MODE = false;
    
    // Columns : Names
    HashMap<String,Integer> Columns_NameToIdx;
    
    // Columns : Indices and positions
    HashMap<Integer,Integer[]> Columns_PosToIdx;
    HashMap<Integer,Integer> Columns_IdxToPos;
    int TableWidth;
    
    // Columns : Data
    ArrayList<Column> Column_list;
    
    // Column data types
    public enum ColumnDataType
    {
        Numeric, Datetime, Levels
    }
    
    public Table()
    {
        Columns_NameToIdx = new HashMap<>();
        
        Columns_PosToIdx = new HashMap<>();
        Columns_IdxToPos = new HashMap<>();
        TableWidth = 0;        
        
        Column_list = new ArrayList<>();
    }
    
    public void AddColumn( String Name, ColumnDataType ColumnType, ArrayList<String> ColumnValues ) 
    {
        Column newCol;
        
        switch ( ColumnType ) 
        {
            case Numeric:
                // Create
                newCol = new Column_Number(Name, ColumnValues);
                
                // Add Column w data
                Column_list.add( newCol );
                
                // Index and position
                Columns_PosToIdx.put( TableWidth, new Integer[]{ Column_list.size() - 1, 0 } );
                Columns_IdxToPos.put( Column_list.size() - 1, TableWidth);
                TableWidth++;
                
            break;
            case Datetime:
                // Create
                newCol = new Column_DateTime(Name, ColumnValues);
                
                // Add Column w data
                Column_list.add( newCol );
                
                // Index and position
                Columns_PosToIdx.put( TableWidth, new Integer[]{ Column_list.size() - 1, 0 } );
                Columns_IdxToPos.put( Column_list.size() - 1, TableWidth);
                TableWidth++;
                
            break;
            case Levels:
                // Create
                newCol = new Column_Levels(Name, ColumnValues);
                
                // Add Column w data
                Column_list.add( newCol );
                                
                // Index and position
                for(int i = 0; i < newCol.GetWidth(); i++ )
                {
                    Columns_PosToIdx.put(TableWidth + i, new Integer[]{ Column_list.size() - 1, i} );
                }
                Columns_IdxToPos.put(Column_list.size() - 1, TableWidth);
                
                TableWidth += newCol.GetWidth();
                
            break;
            default:
                throw new AssertionError();
        }
        
        // Add name
        Columns_NameToIdx.put(Name, Column_list.size() - 1);
        
    }
    
    public void AddColumn( Column new_column ) 
    {
        // Add Column w data
        Column_list.add( new_column );

        // Index and position
        for(int i = 0; i < new_column.GetWidth(); i++ )
        {
            Columns_PosToIdx.put(TableWidth + i, new Integer[]{ Column_list.size() - 1, i} );
        }
        Columns_IdxToPos.put(Column_list.size() - 1, TableWidth);
        
        TableWidth += new_column.GetWidth();
        
        // Add name
        Columns_NameToIdx.put(new_column.GetName(), Column_list.size() - 1);
        
    }
    
    public void SetCellValue( int position, int row, double value )
    {
        Integer[] ColumnIdx = Columns_PosToIdx.get(position);
        Column_list.get( ColumnIdx[0] ).SetValue_Numeric(row, ColumnIdx[1], value);
    }
    
    public double GetCellValueNumeric( int position, int row )
    {
        Integer[] ColumnIdx = Columns_PosToIdx.get(position);
        return Column_list.get( ColumnIdx[0] ).GetValue_Numeric(row, ColumnIdx[1]);
    }
    
    public double GetCellValueNormalized( int position, int row )
    {
        Integer[] ColumnIdx = Columns_PosToIdx.get(position);
        return Column_list.get( ColumnIdx[0] ).GetValue_Normalized(row, ColumnIdx[1]);
    }
    
    public double DenormalizeColumnValue( int position, double value)
    {
        Integer[] ColumnIdx = Columns_PosToIdx.get(position);
        return Column_list.get( ColumnIdx[0] ).DenormalizeValue(value);
    }
    
    public int GetPositionByColumnName(String columnName)
    {
        int columnIdx = Columns_NameToIdx.get(columnName);
        return Columns_IdxToPos.get(columnIdx);
    }
    
    public String GetColumnNameByPosition(int position)
    {
        Integer[] ColumnIdx = Columns_PosToIdx.get(position);
        return Column_list.get( ColumnIdx[0] ).GetName();
    }
    
    public int GetTableWidth()
    {
        return TableWidth;
    }
    
    public int GetRowCount()
    {
        return Column_list.get(0).GetRowCount();
    }
    
    public Double[] GetPositionRange(int position)
    {
        Integer[] ColumnIdx = Columns_PosToIdx.get(position);
        return new Double[] { Column_list.get(ColumnIdx[0]).GetValue_Min() , Column_list.get(ColumnIdx[0]).GetValue_Max() };
    }
    
    public double GetColumnAverage(int position)
    {
        Integer[] ColumnIdx = Columns_PosToIdx.get(position);
        return Column_list.get( ColumnIdx[0] ).GetValue_Avg( ColumnIdx[1] );
    }
    
    public double GetColumnAverageNormalized(int position)
    {
        Integer[] ColumnIdx = Columns_PosToIdx.get(position);
        double avg = Column_list.get( ColumnIdx[0] ).GetValue_Avg( ColumnIdx[1] );
        return Column_list.get( ColumnIdx[0] ).NormalizeValue(avg);
    }
    
    public Double[][] GetNormalizedDataSetAsArray_cr()
    {
        Double[][] DataSet_Norm = new Double[TableWidth][GetRowCount()];
        
        for(int c = 0; c < TableWidth; c++)
        {
            Integer[] ColumnIdx = Columns_PosToIdx.get(c);
            Column col = Column_list.get( ColumnIdx[0] );
            
            DataSet_Norm[c] = col.GetValues_Normalized( ColumnIdx[1] );
        }
        
        return DataSet_Norm;
    }
    
    public double[][] GetNormalizedDataSetAsArray_rc()
    {
        int row_count = GetRowCount();
        double[][] DataSet_Norm = new double[row_count][TableWidth];
        
        for(int c = 0; c < TableWidth; c++)
        {
            Integer[] ColumnIdx = Columns_PosToIdx.get(c);
            Column col = Column_list.get( ColumnIdx[0] );
            
            for(int r = 0; r < row_count; r++)
            {
                Double val = col.GetValue_Normalized( r, ColumnIdx[1] );
                DataSet_Norm[r][c] = (val.isNaN() ? 0d : val);
            }
            
        }
        
        return DataSet_Norm;
    }
    
        
    public String[] ToCSVLines( boolean[] columnMap )
    {
        int rowCount = GetRowCount();
        String[] lines = new String[rowCount+1];
        
        // setup map
        if(columnMap == null)
        {
            columnMap = new boolean[Column_list.size()];
            Arrays.fill( columnMap, true);
        }
        
        // Build header
        StringBuilder Header = new StringBuilder();
        
        int outCol = 0;
        for( int c = 0; c < Math.min(Column_list.size(), columnMap.length); c++)
        {
            if(!columnMap[c])
                continue;
            
            if(outCol > 0)
                Header.append(",");
            
            
            Header.append("\"");
            Header.append(Column_list.get(c).GetName());
            Header.append("\"");
            
            outCol++;
            
        }
        
        lines[0] = Header.toString();
        
        // Build lines
        StringBuilder line;
                
        for(int r = 0; r < rowCount; r++)
        {
            line = new StringBuilder();
            outCol = 0;
            
            for( int c = 0; c < Math.min(Column_list.size(), columnMap.length); c++)
            {
                if(!columnMap[c])
                    continue;

                if(outCol > 0)
                    line.append(",");

                String value = Column_list.get(c).GetValue_String(r);
                                
                if( Column_list.get(c).GetType() == ColumnDataType.Levels)
                {
                    line.append("\"");
                    line.append( value ); 
                    line.append("\"");
                }
                else
                {
                    line.append( value );
                }
                
                outCol++;

            }
            lines[r+1] = line.toString();
            
        }
                
        return lines;
        
    }
    
    public String[] ToCSVLines_norm_data( boolean[] columnMap )
    {
        int rowCount = GetRowCount();
        String[] lines = new String[rowCount+1];
        
        // setup map
        if(columnMap == null)
        {
            columnMap = new boolean[TableWidth];
            Arrays.fill( columnMap, true);
        }
        
        // Build header
        StringBuilder Header = new StringBuilder();
        
        int outCol = 0;
        int subCol = 0;
        String column_name_last = "";
        
        for( int c = 0; c < TableWidth; c++)
        {
            if(!columnMap[c])
                continue;
            
            if(outCol > 0)
                Header.append(",");
            
            String column_name = GetColumnNameByPosition(outCol);
            
            if(column_name.equals( column_name_last ) )
                subCol++;
            else
                subCol = 0;
            
            Header.append("\"");
            if(subCol == 0)
                Header.append( column_name );
            else
            {
                Header.append( column_name );
                Header.append( "." );
                Header.append( subCol );
            }
            Header.append("\"");
            
            outCol++;
            column_name_last = column_name;
        }
        
        lines[0] = Header.toString();
        
        // Build lines
        StringBuilder line;
                
        for(int r = 0; r < rowCount; r++)
        {
            line = new StringBuilder();
            outCol = 0;
            
            for( int c = 0; c < TableWidth; c++)
            {
                if(!columnMap[c])
                    continue;

                if(outCol > 0)
                    line.append(",");

                String value = "" + GetCellValueNormalized(c, r);
                                
                line.append( value );
                
                outCol++;

            }
            lines[r+1] = line.toString();
            
        }
                
        return lines;
        
    }
        
    /**
     * Calculate top [amount] Principal Components of normalised data [0:1].
     * @param tolerance value if average PC value convergence is below, then stop. (~0.00001)
     * @param amount number of PC's to generate. (1 : table width)
     * @return list of top PC's
     */
    public ArrayList<double[]> GetPCs(double tolerance, int amount)
    {
        // Calculate n PC's by NIPALS method.
        
        ArrayList<double[]> pcs = new ArrayList<>();
        
        Double[] x = new Double[TableWidth];
        
        int max_iter = 1000;
        int row_count = GetRowCount();
        
        ArrayList<double[]> loadings_def = new ArrayList<>();
        ArrayList<double[]> scores_def = new ArrayList<>();
        
        double loadings[] = new double[TableWidth];
        double loadings_len;
        double loadings_last[] = new double[TableWidth];
        double scores[] = new double[row_count];
        double scores_len;
        double scores_last[] = new double[row_count];
        //double delta[] = new double[row_count];
        //double delta_len;
        //double delta_len_last = 0d;
        double delta_loadings;
        double delta_loadings_last = 1d;
        
        // for each PC
        for(int a = 0; a < amount; a++)
        {
            // Initialize random vector of width: row count, length: 1
            for(int i = 0; i < scores.length; i++)
                //scores[i] = Math.random();
                //scores[i] = 0.5d;
                scores[i] = GetCellValueNormalized( 0 , i );

            scores_len = 0;
            for (int c = 0; c < scores.length; c++)
                scores_len = scores[c] * scores[c];
            scores_len = Math.pow(scores_len, 0.5 );

            for(int i = 0; i < scores.length; i++)
                scores[i] = scores[i] / scores_len;
            
            // power iterate
            boolean stop = false;
            int iterations = 0;
            
            //System.out.println("\tpc : " + (a+1));
            
            while(!stop)
            {
                // Save old loadings
                for(int c = 0; c < TableWidth; c++)
                    loadings_last[c] = loadings[c];
                
                // loadings: p = X't/t't  
                // X't
                Arrays.fill(loadings, 0d);
                for(int r = 0; r < row_count; r++)
                {
                    for(int c = 0; c < TableWidth; c++)
                    {
                        // Retrieve data point, if NaN -> 0
                        x[c] = GetCellValueNormalized( c, r);
                        if(x[c].isNaN())
                            x[c] = 0d;

                        // Deflate data with old pc's
                        // x = x - (th %*% t(ph))
                        for(int d = 0; d < a; d++)
                            x[c] = x[c] - ( loadings_def.get(d)[c] * scores_def.get(d)[r] );

                        //  if( c == 4 && r == 7) System.out.println("\ta" + d + " i" + iterations + "\tp:" + loadings_def.get(d)[c] + ", t:"+ scores_def.get(d)[r] + "\tx:" + x[c]);
                    }
                    
                    for(int c = 0; c < TableWidth; c++)
                        loadings[c] += x[c] * scores[r];
                    
                }
                
                // 1 / t't
                scores_len = 0d;
                for(int c = 0; c < row_count; c++)
                    scores_len += scores[c] * scores[c];
                
                for(int c = 0; c < TableWidth; c++)
                {
                    loadings[c] += loadings[c] / scores_len;
                }
                
                
                // Normalize loadings
                loadings_len = 0d;
                for(int c = 0; c < TableWidth; c++)
                    loadings_len += loadings[c] * loadings[c];
                loadings_len = Math.pow( loadings_len, 0.5d );
                
                for(int c = 0; c < TableWidth; c++)
                    loadings[c] = loadings[c] / loadings_len;
                
                // Save scores
                for(int r = 0; r < row_count; r++)
                    scores_last[r] += scores[r];
                
                // Scores : th = x %*% ph / sum(ph*ph)
                Arrays.fill(scores, 0d);
                for(int r = 0; r < row_count; r++)
                {
                    for(int c = 0; c < TableWidth; c++)
                    {
                        // Retrieve data point, if NaN -> 0
                        x[c] = GetCellValueNormalized( c, r);
                        if(x[c].isNaN())
                            x[c] = 0d;
                        
                        // Deflate data with old pc's
                        // x = x - (th %*% t(ph))
                        for(int d = 0; d < loadings_def.size(); d++)
                            x[c] = x[c] - ( loadings_def.get(d)[c] * scores_def.get(d)[r] );
                    }
                    
                    for(int c = 0; c < TableWidth; c++)
                        scores[r] += x[c] * loadings[c];
                    
                }
                                
                /*
                // Check convergence of scores
                for(int r = 0; r < row_count; r++)
                    delta[r] = scores[r] - scores_last[r];
                delta_len = 0d;
                for(int r = 0; r < row_count; r++)
                    delta_len += delta[r] * delta[r];
                
                delta_len = 1 / Math.pow( delta_len, 0.5 );
                */
                
                // Check convergence of loadings
                delta_loadings = 0d;
                for(int c = 0; c < TableWidth; c++)
                    delta_loadings += Math.abs(loadings[c] - loadings_last[c]);
                delta_loadings = delta_loadings / (double) TableWidth;
               
                //System.out.println("\tPC" + a + "\tIter: " + iterations + "\tconvergence: " + delta_loadings + "\tload0:" + loadings[0]);
                
                // Stop if convergence is too low, if zero convergence is happening, or max iterations is reached
                if( delta_loadings < tolerance || delta_loadings == delta_loadings_last || iterations > max_iter)
                    stop = true;
                
                delta_loadings_last = delta_loadings;
                //delta_len_last = delta_len;
                iterations++;
                                
            }
            
            // Save PC
            pcs.add( Arrays.copyOf(loadings, TableWidth) );
            
            // Deflate variation by PC, save loadings and scores
            // x <- x - (th %*% t(ph))
            loadings_def.add( Arrays.copyOf(loadings, TableWidth) );
            scores_def.add( Arrays.copyOf(scores, row_count) );
            
        }
        
        return pcs;
        
    }
    
    
}


