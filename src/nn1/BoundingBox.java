/*
 * The MIT License
 *
 * Copyright 2018 JCM.
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

/**
 *
 */
public abstract class BoundingBox
{
    public double[] Box_origin_point;
    public double[] Box_end_point;

    public int Coords_origin_x;
    public int Coords_origin_y;
    public int Coords_end_x;
    public int Coords_end_y;

    RTree_3d_ref Tree;
    BoundingBoxNode ParentBox;
    int Depth;
    
    double[][][] PointMatrix;

    int Capacity_min = 2;
    int Capacity_max = 10;

    int Dimensions;
    boolean IsLeaf;

    //<editor-fold defaultstate="collapsed" desc="Hypervolume calculations">
    public double GetHyperVolume()
    {
        double hyper_vol = 0d;
        for(int a = 0; a < Box_origin_point.length; a++)
        {
            hyper_vol += Math.log10(Box_end_point[a] - Box_origin_point[a]);
        }
        return hyper_vol;
    }

    public double GetHyperVolume( double[] item )
    {
        double hyper_vol = 0d;
        for(int a = 0; a < Box_origin_point.length; a++)
        {
            if( Box_origin_point[a] > item[a])
                hyper_vol += Math.log10( Box_end_point[a] - item[a] );
            else if(Box_end_point[a] < item[a] )
                hyper_vol += Math.log10( item[a] - Box_origin_point[a] );
            else
                hyper_vol += Math.log10( Box_end_point[a] - Box_origin_point[a] );
        }
        return hyper_vol;
    }

    public double GetHyperVolume( BoundingBox box )
    {
        double hyper_vol = 0d;
        for(int a = 0; a < Box_origin_point.length; a++)
        {
            if( Box_origin_point[a] < box.Box_origin_point[a])
                if( Box_end_point[a] > box.Box_end_point[a])
                    hyper_vol += Math.log10( (Box_end_point[a] - Box_origin_point[a]) );
                else
                    hyper_vol += Math.log10( ( box.Box_end_point[a] - Box_origin_point[a]) );
            else
                if( Box_end_point[a] > box.Box_end_point[a])
                    hyper_vol += Math.log10( ( Box_end_point[a] - box.Box_origin_point[a]) );
                else
                    hyper_vol += Math.log10( ( box.Box_end_point[a] - box.Box_origin_point[a]) );
        }
        return 2 * hyper_vol;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Box and Coords">
    public void ExpandCoordsToFit( int x, int y )
    {
        if(Coords_origin_x > x)
            Coords_origin_x = x;
        if(Coords_origin_y > y)
            Coords_origin_y = y;
        if(Coords_end_x < x)
            Coords_end_x = x;
        if(Coords_end_y < y)
            Coords_end_y = y;
    }
    
    public void SetCoordsToFit( int x, int y )
    {
        Coords_origin_x = x;
        Coords_origin_y = y;
        Coords_end_x = x;
        Coords_end_y = y;
    }

    public void ExpandCoordsToFit( BoundingBox box )
    {
        if(Coords_origin_x > box.Coords_origin_x)
            Coords_origin_x = box.Coords_origin_x;
        if(Coords_origin_y > box.Coords_origin_y)
            Coords_origin_y = box.Coords_origin_y;
        if(Coords_end_x < box.Coords_end_x)
            Coords_end_x = box.Coords_end_x;
        if(Coords_end_y < box.Coords_end_y)
            Coords_end_y = box.Coords_end_y;
    }

    public void ExpandBoxToFit( double[] item )
    {
        for(int a = 0; a < Box_origin_point.length; a++)
        {
            if( Box_origin_point[a] > item[a])
                Box_origin_point[a] = item[a];

            if(Box_end_point[a] < item[a] )
                Box_end_point[a] = item[a];
        }
    }

    public void ExpandBoxToFit( BoundingBox box )
    {
        for(int a = 0; a < Box_origin_point.length; a++)
        {
            if( Box_origin_point[a] > box.Box_origin_point[a])
                Box_origin_point[a] = box.Box_origin_point[a];

            if(Box_end_point[a] < box.Box_end_point[a] )
                Box_end_point[a] = box.Box_end_point[a];
        }
    }

    public void SetBoxToFit( BoundingBox box )
    {
        for(int a = 0; a < Box_origin_point.length; a++)
        {
            Box_origin_point[a] = box.Box_origin_point[a];
            Box_end_point[a] = box.Box_end_point[a];
        }
    }
    
    public void SetBoxToFit( double[] item )
    {
        for(int a = 0; a < Box_origin_point.length; a++)
        {
            Box_origin_point[a] = item[a];
            Box_end_point[a] = item[a];
        }
    }

    abstract protected void RefreshBox();
    
    public boolean EnclosesPoint( double[] point )
    {
        for(int a = 0; a < Box_origin_point.length; a++)
            if( Box_origin_point[a] > point[a] || Box_end_point[a] < point[a] )
                return false;

        return true;
    }

    public boolean EnclosesPoint( int x, int y )
    {
        if( Coords_origin_x > x || Coords_end_x < x )
            return false;

        if( Coords_origin_y > y || Coords_end_y < y )
            return false;

        return true;
    }

    public double DistanceToBox( double[] point )
    {
        double distance = 0d;
        for(int a = 0; a < Box_origin_point.length; a++)
        {
            if( Box_origin_point[a] > point[a])
                distance += Math.pow(Box_origin_point[a] - point[a], 2);
            else if( Box_end_point[a] < point[a])
                distance += Math.pow(point[a] - Box_end_point[a], 2);
        }

        return Math.pow( distance, 0.5d);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Insert, remove, update points">

    abstract void InsertPoint( int x, int y );
    abstract ArrayList<int[]> Split();
    abstract void RefreshPoint( int x, int y );
    abstract boolean RemovePoint( int x, int y );
    
    abstract boolean IsAboveMaxCapacity();
    abstract boolean IsBelowMinCapacity();
    
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Object override">
    
    @Override
    public String toString()
    {
        String desc = "";
        if(this.IsLeaf)
            desc += "BoundingBox{Leaf}";
        else
            desc += "BoundingBox{Leaf}";

        for(int i = 0; i < 3; i++)
            desc += Box_origin_point[i] + ",";

        return desc;
    }
    
    //</editor-fold>
}