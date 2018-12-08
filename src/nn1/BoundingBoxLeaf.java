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

public class BoundingBoxLeaf extends BoundingBox
{
    //<editor-fold defaultstate="collapsed" desc="Contstructor and vars">

    ArrayList<int[]> Items; 

    public BoundingBoxLeaf( double[][][] points, BoundingBoxNode parent, RTree_3d_ref tree, int depth )
    {
        this( points, parent, tree, depth, -1, -1 );
    }
    
    public BoundingBoxLeaf( double[][][] points, BoundingBoxNode parent, RTree_3d_ref tree, int depth, int item_x, int item_y )
    {
        IsLeaf = true;
        Depth = depth;
        Tree = tree;
        ParentBox = parent;
        PointMatrix = points;
        Items = new ArrayList<>();
        Dimensions = points[0][0].length;

        Box_origin_point = new double[Dimensions];
        Box_end_point = new double[Dimensions];
        
        if( item_x > 0 && item_y > 0)
        {
            for(int a = 0; a < Box_origin_point.length; a++)
            {
                Box_origin_point[a] = PointMatrix[item_x][item_y][a];
                Box_end_point[a] = PointMatrix[item_x][item_y][a];
            }

            Items.add( new int[]{ item_x, item_y } );
        }

    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Insert and split">

    @Override
    public void InsertPoint( int x, int y )
    {
        // Expand box
        ExpandBoxToFit( PointMatrix[x][y] );

        // Add item
        Items.add( new int[] { x, y } );

        // Check for split
        if( Items.size() > Capacity_max)
            Split();
    }


    @Override
    public ArrayList<int[]> Split()
    {
        // Find highest distance points
        double hypervolume_best = -Double.MAX_VALUE;
        int hypervolume_best_i = 0;
        int hypervolume_best_j = 0;
        double hypervolume = 0d;

        for( int i = 0; i < Capacity_max + 1; i++)
        {
            for( int j = i + 1; j < Capacity_max + 1; j++)
            {
                hypervolume = 0d;
                double[] point1 = PointMatrix[ Items.get(i)[0] ][ Items.get(i)[1] ];
                double[] point2 = PointMatrix[ Items.get(j)[0] ][ Items.get(j)[1] ];

                for(int a = 0; a < Dimensions; a++)
                {
                    if( point1[a] > point2[a] )
                        hypervolume += Math.log10(point1[a] - point2[a]);
                    if( point1[a] < point2[a] )
                        hypervolume += Math.log10(point2[a] - point1[a]);
                }

                if( hypervolume_best < hypervolume)
                {
                    hypervolume_best_i = i;
                    hypervolume_best_j = j;
                    hypervolume_best = hypervolume;
                }
            }
        }

        // Ready the split
        ArrayList<int[]> items_split = new ArrayList<>();
        items_split.addAll(Items);

        int point1_x = items_split.get(hypervolume_best_i)[0];
        int point1_y = items_split.get(hypervolume_best_i)[1];
        int point2_x = items_split.get(hypervolume_best_j)[0];
        int point2_y = items_split.get(hypervolume_best_j)[1];
        double[] point1 = PointMatrix[ point1_x ][ point1_y ];
        double[] point2 = PointMatrix[ point2_x ][ point2_y ];

        if(hypervolume_best_i > hypervolume_best_j)
        {
            items_split.remove(hypervolume_best_i);
            items_split.remove(hypervolume_best_j);
        }
        else
        {
            items_split.remove(hypervolume_best_j);
            items_split.remove(hypervolume_best_i);
        }

        // Create new box with point1
        BoundingBoxLeaf box1 = new BoundingBoxLeaf( PointMatrix, ParentBox, Tree, Depth, point1_x, point1_y );

        box1.SetBoxToFit(point1);
        box1.SetCoordsToFit( point1_x, point1_y);
        
        // Reset this box, and add point2
        BoundingBoxLeaf box2 = this;
        
        box2.SetBoxToFit(point2);
        box2.SetCoordsToFit( point2_x, point2_y);

        box2.Items.clear();
        box2.Items.add( new int[]{ point2_x, point2_y } );

        // Notify parent of split
        ParentBox.InsertBox_internal(box1);
        
        // Add items to the new boxes, up to min capacity
        int[] item_best;
        
        // box1
        while( box1.IsBelowMinCapacity() && !items_split.isEmpty() )
        {
            hypervolume_best = Double.MAX_VALUE;
            item_best = null;
            int index_best = -1;
            
            for(int i = 0; i < items_split.size(); i++ )
            {
                int[] item = items_split.get(i);
                double[] point = PointMatrix[ item[0] ][ item[1] ];
                
                hypervolume = box1.GetHyperVolume( point );
                
                if(hypervolume_best > hypervolume)
                {
                    hypervolume_best = hypervolume;
                    item_best = item; 
                    index_best = i;
                }
                
            }
            
            if(item_best != null)
            {
                box1.Items.add( new int[]{ item_best[0], item_best[1] } );
                box1.ExpandBoxToFit( PointMatrix[ item_best[0] ][ item_best[1] ] );
                box1.ExpandCoordsToFit( item_best[0], item_best[1]);
                
                items_split.remove(index_best);
            }
        }
        
        // box2
        while( box2.IsBelowMinCapacity() && !items_split.isEmpty() )
        {
            hypervolume_best = Double.MAX_VALUE;
            item_best = null;
            int index_best = -1;
            
            for(int i = 0; i < items_split.size(); i++ )
            {
                int[] item = items_split.get(i);
                double[] point = PointMatrix[ item[0] ][ item[1] ];
                hypervolume = box1.GetHyperVolume( point );
                
                if(hypervolume_best > hypervolume)
                {
                    hypervolume_best = hypervolume;
                    item_best = item; 
                    index_best = i;
                }
                
            }
            
            if(item_best != null)
            {
                box2.Items.add( new int[]{ item_best[0], item_best[1] } );
                box2.ExpandBoxToFit( PointMatrix[ item_best[0] ][ item_best[1] ] );
                box2.ExpandCoordsToFit( item_best[0], item_best[1]);
                
                items_split.remove(index_best);
            }
        }
        
        // return remaining to be reinserted into the tree
        return items_split;        
    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Remove point">

    @Override
    public boolean RemovePoint( int x, int y )
    {
        boolean success = false;

        // Remove from list
        for(int i = 0; i < Items.size(); i++)
            if( Items.get(i)[0] == x && Items.get(i)[1] == y )
            {
                Items.remove(i);
                success = true;
                break;
            }

        return success;
    }

    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Misc">

    @Override
    protected void RefreshBox()
    {
        // Recalculate corners
        int[] item0_coords = Items.get(0);
        double[] item0 = PointMatrix[item0_coords[0]][item0_coords[1]];
        for(int a = 0; a < Dimensions; a++)
        {
            Box_origin_point[a] = item0[a];
            Box_end_point[a] = item0[a];
        }
        for(int i = 0; i < Items.size(); i++)
        {
            int[] item_coords = Items.get(i);
            double[] item = PointMatrix[item_coords[0]][item_coords[1]];

            for(int a = 0; a < Dimensions; a++)
            {
                if(Box_origin_point[a] > item[a])
                    Box_origin_point[a] = item[a];
                if(Box_end_point[a] < item[a])
                    Box_end_point[a] = item[a];
            }

            if(Coords_origin_x > item_coords[0])
                Coords_origin_x = item_coords[0];

            if(Coords_origin_y > item_coords[1])
                Coords_origin_y = item_coords[1];

            if(Coords_end_x < item_coords[0])
                Coords_end_x = item_coords[0];

            if(Coords_end_y < item_coords[1])
                Coords_end_y = item_coords[1];

        }

        // Refresh ancestors
        BoundingBoxNode box_ancestor = ParentBox;
        while(box_ancestor != null)
        {
            box_ancestor.RefreshBox();
            box_ancestor = box_ancestor.ParentBox;
        }
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
}
    