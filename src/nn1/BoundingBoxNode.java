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

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.TreeSet;

/**
 *
 */
public class BoundingBoxNode extends BoundingBox
{
    ArrayList<BoundingBox> Items;
    
    public BoundingBoxNode( double[][][] points, BoundingBoxNode parent, BoundingBox firstBox, RTree_3d_ref tree, int depth )
    {
        Depth = depth;
        Tree = tree;
        PointMatrix = points;
        Items = new ArrayList<>();
        ParentBox = parent;
        Dimensions = points[0][0].length;

        Box_origin_point = new double[Dimensions];
        Box_end_point = new double[Dimensions];

        if(firstBox != null)
        {
            for(int a = 0; a < Box_origin_point.length; a++)
            {
                Box_origin_point[a] = firstBox.Box_origin_point[a];
                Box_end_point[a] = firstBox.Box_end_point[a];
            }

            Items.add( firstBox );
        }
        else
        {
            for(int a = 0; a < Box_origin_point.length; a++)
            {
                Box_origin_point[a] = Double.MAX_VALUE;
                Box_end_point[a] = Double.MIN_VALUE;
            }
        }

        IsLeaf = false;
    }

    //<editor-fold defaultstate="collapsed" desc="Box and Coords">
    
    @Override
    protected void RefreshBox()
    {
        // Recalculate corners
        BoundingBox box0 = Items.get(0);
        for(int a = 0; a < Dimensions; a++)
        {
            Box_origin_point[a] = box0.Box_origin_point[a];
            Box_end_point[a] = box0.Box_end_point[a];
        }
        for(int i = 0; i < Items.size(); i++)
        {
            BoundingBox box = Items.get(i);

            for(int a = 0; a < Dimensions; a++)
            {
                if(Box_origin_point[a] > box.Box_origin_point[a])
                    Box_origin_point[a] = box.Box_origin_point[a];
                if(Box_end_point[a] < box.Box_end_point[a])
                    Box_end_point[a] = box.Box_end_point[a];
            }

            if(Coords_origin_x > box.Coords_origin_x)
                Coords_origin_x = box.Coords_origin_x;

            if(Coords_origin_y > box.Coords_origin_y)
                Coords_origin_y = box.Coords_origin_y;

            if(Coords_end_x < box.Coords_end_x)
                Coords_end_x = box.Coords_end_x;

            if(Coords_end_y < box.Coords_end_y)
                Coords_end_y = box.Coords_end_y;
        }
    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Insert and split">

    @Override
    public void InsertPoint( int x, int y )
    {
        // Expand box
        ExpandBoxToFit( PointMatrix[x][y] );
        ExpandCoordsToFit( x, y );

        if(Items.isEmpty())
        {
            BoundingBoxLeaf leaf = new BoundingBoxLeaf(PointMatrix, this, Tree, Depth + 1, x, y);
            Items.add( leaf );
            return;
        }

        // Search sub-box'es
        for(int i = 0; i < Items.size(); i++)
        {
            if( Items.get(i).EnclosesPoint( PointMatrix[x][y] ) )
            {
                Items.get(i).InsertPoint(x, y);
                return;
            }
        }

        // No box encloses point
        // Find one to expand
        BoundingBox bestBox = Items.get(0);
        double best_volume = bestBox.GetHyperVolume( PointMatrix[x][y] );

        for(int i = 1; i < Items.size(); i++)
        {
            double volume = Items.get(i).GetHyperVolume( PointMatrix[x][y] );
            if( volume < best_volume )
            {
                best_volume = volume;
                bestBox = Items.get(i);
            }
        }

        // Insert item
        bestBox.InsertPoint(x, y);
    }


    public void InsertBox_external( BoundingBox box )
    {
        Items.add(box);

        ExpandBoxToFit(box);
        ExpandCoordsToFit(box);

        if(Items.size() > Capacity_max)
            Split();
    }

    public void InsertBox_internal( BoundingBox box )
    {
        Items.add(box);
    }


    @Override
    public ArrayList<int[]> Split()
    {
        int items_size = Items.size();

        double hypervol;
        double best_volumne = Double.MAX_VALUE;
        int best_volume_i = 0;
        int best_volume_j = 0;

        // Find lowest vol
        for( int i = 0; i < items_size; i++)
            for( int j = i+1; j < items_size; j++)
            {
                hypervol = Items.get(i).GetHyperVolume( Items.get(j) );
                if(hypervol < best_volumne)
                {
                    best_volumne = hypervol;
                    best_volume_i = i;
                    best_volume_j = j;
                }
            }

        // Ready the split
        ArrayList<BoundingBox> items_split = new ArrayList<>();
        items_split.addAll(Items);

        // Create new box
        BoundingBoxNode box1 = new BoundingBoxNode( PointMatrix, ParentBox, Items.get(best_volume_i), Tree, Depth + 1 );

        // Reset this box
        BoundingBoxNode box2 = new BoundingBoxNode( PointMatrix, ParentBox, Items.get(best_volume_j), Tree, Depth + 1 );

        Items.clear();
        Items.add( box1 );
        Items.add( box2 );

        // Remove the two initial boxes from the list
        if(best_volume_i > best_volume_j)
        {
            items_split.remove(best_volume_i);
            items_split.remove(best_volume_j);
        }
        else
        {
            items_split.remove(best_volume_j);
            items_split.remove(best_volume_i);
        }
        
        // distribute the remaining boxes
        for(int i = 0; i < items_split.size(); i++ )
        {
            BoundingBox item = items_split.get(i);

            if( box1.GetHyperVolume( item ) > box2.GetHyperVolume( item ) )
                box2.InsertBox_external(item );
            else
                box1.InsertBox_external( item );

        }

        SetBoxToFit( box1 );
        ExpandBoxToFit( box2 );

        // Notify parent of split
        if( ParentBox != null)
            ParentBox.InsertBox_internal(box1);
    }

    @Override
    public void RefreshPoint( int x, int y)
    {

    }
    
    @Override
    public boolean IsAboveMaxCapacity()
    {
        return Items.size() > Capacity_max;
    }
    
    @Override
    public boolean IsBelowMinCapacity()
    {
        return Items.size() < Capacity_min;
    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Remove point">

    @Override
    public BoundingBox RemovePoint( int x, int y )
    {
        double[] point = PointMatrix[x][y];
        BoundingBox box_pointRemoved;

        for(int i = 0; i < Items.size(); i++)
        {
            BoundingBox box = Items.get(i);

            if(box.EnclosesPoint(point))
            {
                box_pointRemoved = box.RemovePoint(x, y);
                if(box_pointRemoved != null)
                {
                    RefreshBox();
                    return box_pointRemoved;
                }
            }
        }

        return null;
    }

    //</editor-fold>
    
    
    
}