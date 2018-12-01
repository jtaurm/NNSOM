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

import java.io.*;
import java.util.*;
import java.time.*;
import java.time.format.*;

/**
 *
 */
public class CsvReader 
{
    public static final boolean DEV_MODE = false;
    
    private static List<List<String>> ReadAsStringNestedLists(String fileName, String seperator)
    {
        File file = new File(fileName);

        // this gives you a 2-dimensional array of strings
        List<List<String>> lines = new ArrayList<>();
        Scanner inputStream;

        try
        {
            inputStream = new Scanner(file);

            while(inputStream.hasNext())
            {
                //String line = inputStream.next();
                String line = inputStream.nextLine();
                String[] values = line.split(seperator);
                lines.add(Arrays.asList(values));
            }

            inputStream.close();
            
        }catch (FileNotFoundException e) 
        {
            e.printStackTrace();
        }
        
        return lines;
    }
    
    public static Table ReadAllAsDouble(String fileName, String seperator, boolean hasHeader)
    {
        if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Reading file");
        
        // Read all as strings
        List<List<String>> lines = ReadAsStringNestedLists(fileName, seperator);
        
        if(Nn1.DEV_MODE && DEV_MODE) System.out.println("File has " + lines.size() + " rows.");
        if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Header has " + lines.get(1).size() + " cols.");
        if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Data has " + lines.get(2).size() + " cols.");
        
        // Get column names
        ArrayList<String> ColumnNames = new ArrayList<>();
        
        List<String> header = lines.get(0);
        for(int i = 0; i < header.size(); i++)
        {
            String value = header.get(i);
            
            if( value.startsWith("\"") && value.endsWith("\"") ) // remove quotes
                value = value.substring( 1, value.length() - 1 );
            
            ColumnNames.add(value);
        }
        
        // Create empty structure for column storing
        ArrayList<ArrayList<String>> columnData = new ArrayList<>();
        
        for(int c = 0; c < ColumnNames.size(); c++)
        {
            columnData.add( new ArrayList<String>() );
        }
        
        // Convert from string rows into column arrays
        if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Parsing data");
        
        for(int l = 0; l < lines.size(); l++)
        {
            List<String> line = lines.get(l);
            
            if( l == 0) // Skip header
                continue;
            
            for(int c = 0; c < ColumnNames.size(); c++)
            {
                String value = line.get(c);
                
                if( value.startsWith("\"") && value.endsWith("\"") )
                    value = value.substring( 1, value.length() - 1 );
                
                columnData.get(c).add(value);
            }
        }
        
        // Fill table
        if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Generating table structure (" + ColumnNames.size() + " cols)");
        
        Table tCSV = new Table();
        
        for(int c = 0; c < ColumnNames.size(); c++)
        {
            if( Column_Number.Validator.ValidateType( columnData.get(c) ) )
            {
                if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Column "+ ColumnNames.get(c) + " is Numeric ");
                tCSV.AddColumn( ColumnNames.get(c), Table.ColumnDataType.Numeric, columnData.get(c) );
            }
            else if( Column_DateTime.Validator.ValidateType( columnData.get(c) ) )
            {
                if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Column "+ ColumnNames.get(c) + " is DateTime ");
                tCSV.AddColumn( ColumnNames.get(c), Table.ColumnDataType.Datetime, columnData.get(c) );
            }
            else
            {
                if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Column "+ ColumnNames.get(c) + " is Levels ");
                tCSV.AddColumn( ColumnNames.get(c), Table.ColumnDataType.Levels, columnData.get(c) );
            }
        }
        
        if(Nn1.DEV_MODE && DEV_MODE) System.out.println("Table is ready");
        
        return tCSV;
        
    }
    
    public static void WriteLines(String fileName, String[] lines)
    {
        
        BufferedWriter writer = null;
        try 
        {
            //create a temporary file
            File logFile = new File(fileName);
            writer = new BufferedWriter(new FileWriter(logFile));
            
            for(String line : lines)
            {
                writer.write(line);
                writer.write("\n");
            }
            
            
        } catch (Exception e) 
        {
            e.printStackTrace();
        } finally 
        {
            try 
            {
                // Close the writer regardless of what happens...
                writer.close();
            } catch (Exception e) 
            {
            }
        }
        
    }
}
