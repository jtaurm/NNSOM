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
 * @author JCM
 */
public class RTree_3d_ref 
{
    double[][][] PointMatrix;
    
    int Dimensions;
    
    BoundingBoxNode RootNode;
    
    public RTree_3d_ref(double[][][] points)
    {
        PointMatrix = points;
        Dimensions = points[0][0].length;
        RootNode = new BoundingBoxNode( points, null, null, this, 0);
    }
    
    public void Insert(int x, int y)
    {
        double[] point = PointMatrix[x][y];
        
        BoundingBox box = RootNode;
        BoundingBox box_best;
        BoundingBoxNode box_node;
        
        ArrayList<BoundingBox> box_line = new ArrayList<>();
        
        // Find ancestral line of insertion box
        while( !box.IsLeaf )
        {
            box_node = (BoundingBoxNode) box;
            box_best = null;
            ArrayList<BoundingBox> Items = box_node.Items;
            
            // Find box enclosing point, if any
            for(int i = 0; i < Items.size(); i++)
            {
                if( Items.get(i).EnclosesPoint( PointMatrix[x][y] ) )
                {
                    box_best = Items.get(i);
                }
            }
            
            if(box_best == null)
            {
                // No enclosing box found, find the one that expands the least on insert
                double best_volume = box_best.GetHyperVolume( PointMatrix[x][y] );

                for(int i = 1; i < Items.size(); i++)
                {
                    double volume = Items.get(i).GetHyperVolume( PointMatrix[x][y] );
                    if( volume < best_volume )
                    {
                        best_volume = volume;
                        box_best = Items.get(i);
                    }
                }
                
            }
            
            if(box_best == null)
            {
                // There are no sub boxes! Create one
                BoundingBoxLeaf leaf_new = new BoundingBoxLeaf(PointMatrix, box_node, this, box.Depth + 1);
                leaf_new.SetBoxToFit( point );
                leaf_new.SetCoordsToFit(x, y);
                
                box_line.add(leaf_new);
                box = leaf_new;
            }
            else
            {
                // Remember choice
                box_line.add(box_best);
                box = box_best;
            }
            
        }
        
        // Box is leaf, and may be inserted into
        BoundingBoxLeaf box_leaf = (BoundingBoxLeaf) box;
        
        box_leaf.Items.add( new int[] { x, y } );
        
        // Expand ancestral line to fit
        for(int i = box_line.size() - 1; i > -1; i--)
        {
            box_line.get(i).ExpandBoxToFit( point );
            box_line.get(i).ExpandCoordsToFit(x, y);
        }
        
        // Check for splits in the ancestral line
        ArrayList<int[]> reinserts = new ArrayList<>();
        
        for(int i = box_line.size() - 1; i > -1; i--)
        {
            box = box_line.get(i);
            
            if( box.IsAboveMaxCapacity() )
            {
                reinserts.addAll( box.Split() );
            }
        }
        
        // Reinsert recursively
        for(int i = 0; i < reinserts.size(); i++)
            Insert( reinserts.get(i)[0], reinserts.get(i)[1] );
    }
    
    public void Remove(int x, int y)
    {
        BoundingBox box = RootNode;
        BoundingBoxLeaf box_rem = null;
        
        ArrayList<BoundingBox> box_line = new ArrayList<>();
        TreeSet<BoundingBox> queue = new TreeSet<>();
        
        queue.add(RootNode);
        
        // Find ancestral line of insertion box
        while( !queue.isEmpty() )
        {
            box = queue.first();
            queue.remove(box);
                        
            if( box.IsLeaf )
            {
                box_rem = (BoundingBoxLeaf) box;
                
                if( box_rem.RemovePoint(x, y) )
                {   // Found correct leaf and removed point
                    break;
                }
            }
            else
            {
                // Keep looking at children
                BoundingBoxNode box_node = (BoundingBoxNode) box;
                ArrayList<BoundingBox> children = box_node.Items;
                
                // Find box enclosing point, if any
                for(int i = 0; i < children.size(); i++)
                {
                    if( children.get(i).EnclosesPoint( x, y ) )
                    {
                        queue.add( children.get(i) );
                    }
                }
            }
            
        }
        
        // Refers
        box = box_rem;
        while( box != null )
        {
            box.RefreshBox();
            box = box.ParentBox;
        }
        
    }
    
    public void Refresh(int x, int y)
    {
        
    }
    
    public void ReInsert(int x, int y)
    {
        
    }
    
    //<editor-fold defaultstate="collapsed" desc="Search">
    
    private class BoxQueueEntry implements Comparable
    {
        public BoundingBox box;
        public double distance;

        public BoxQueueEntry(double distance, BoundingBox box)
        {
            this.box = box;
            this.distance = distance;
        }

        @Override
        public int compareTo( Object other )
        {
            if( this.distance < ((BoxQueueEntry) other).distance )
                return -1;
            if( this.distance > ((BoxQueueEntry) other).distance )
                return 1;

            for(int i = 0; i < this.box.Dimensions; i++)
            {
                if( this.box.Box_origin_point[i] < ((BoxQueueEntry) other).box.Box_origin_point[i] )
                    return -1;
                if( this.box.Box_origin_point[i] > ((BoxQueueEntry) other).box.Box_origin_point[i] )
                    return 1;

                if( this.box.Box_end_point[i] < ((BoxQueueEntry) other).box.Box_end_point[i] )
                    return -1;
                if( this.box.Box_end_point[i] > ((BoxQueueEntry) other).box.Box_end_point[i] )
                    return 1;
            }

            if( System.identityHashCode(this) < System.identityHashCode(other) )
                return -1;
            if( System.identityHashCode(this) > System.identityHashCode(other) )
                return 1;

            return 0;
        }

        @Override
        public String toString()
        {
            return distance + ":" + box.toString() + "@" + System.identityHashCode(this);
        }

    }

    public int[] FindNearestNeighbour( double[] point )
    {
        // Initialize queue
        TreeSet<BoxQueueEntry> queue = new TreeSet<>();

        for(int i = 0; i < RootNode.Items.size(); i++)
        {
            BoundingBox box = RootNode.Items.get(i);
            queue.add( new BoxQueueEntry( box.DistanceToBox(point), box ) );
        }

        double distance_best = Double.MAX_VALUE;
        int coord_x_best = -1;
        int coord_y_best = -1;


        while(!queue.isEmpty())
        {
            BoxQueueEntry entry = queue.first();
            queue.remove(entry);

            if(entry.distance > distance_best)
                // box is too far away to have the nearest neighbour
                continue;

            if(entry.box.IsLeaf)
            {
                // Leaf - each point in the leaf must be checked
                ArrayList<int[]> items_child = ((BoundingBoxLeaf) entry.box).Items;
                for(int i = 0; i < items_child.size(); i++)
                {
                    // Calculate distance from child to point
                    double distance_child = 0d;
                    int coord_x_child = items_child.get(i)[0];
                    int coord_y_child = items_child.get(i)[1];

                    double[] point_child = PointMatrix[ coord_x_child ][ coord_y_child ];

                    for(int a = 0; a < Dimensions; a++)
                    {
                        distance_child += Math.pow( point_child[a] - point[a], 2d );
                    }
                    distance_child = Math.pow( distance_child, 0.5d );

                    // Keep best
                    if(distance_child < distance_best)
                    {
                        coord_x_best = coord_x_child;
                        coord_y_best = coord_y_child;
                        distance_best = distance_child;
                    }
                }
            }
            else
            {
                // Not leaf - just all all children to queue
                ArrayList<BoundingBox> items_child = ((BoundingBoxNode) entry.box).Items;
                for(int i = 0; i < items_child.size(); i++)
                {
                    BoundingBox box = items_child.get(i);
                    queue.add( new BoxQueueEntry( box.DistanceToBox(point), box ) );
                }
            }

        }

        return new int[]{ coord_x_best, coord_y_best };

    }
    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Debugging">
    public void PrintAll(double[] point)
    {

        NumberFormat format5d = new DecimalFormat("#0.00000");

        // Initialize queue
        TreeSet<BoxQueueEntry> queue = new TreeSet<>();

        for(int i = 0; i < RootNode.Items.size(); i++)
        {
            BoundingBox box = RootNode.Items.get(i);
            queue.add( new BoxQueueEntry( box.DistanceToBox(point), box ) );
        }

        double distance_best = Double.MAX_VALUE;
        int coord_x_best = -1;
        int coord_y_best = -1;

        while(!queue.isEmpty())
        {
            BoxQueueEntry entry = queue.first();
            queue.remove(entry);

            if(entry.distance > distance_best)
                System.out.println("too far");
            else
                System.out.println("check");
                // box is too far away to have the nearest neighbour
            //    continue;

            if(entry.box.IsLeaf)
            {
                // Leaf - each point in the leaf must be checked
                ArrayList<int[]> items_child = ((BoundingBoxLeaf) entry.box).Items;
                for(int i = 0; i < items_child.size(); i++)
                {
                    // Calculate distance from child to point
                    double distance_child = 0d;
                    int coord_x_child = items_child.get(i)[0];
                    int coord_y_child = items_child.get(i)[1];

                    double[] point_child = PointMatrix[ coord_x_child ][ coord_y_child ];

                    for(int a = 0; a < Dimensions; a++)
                    {
                        distance_child += Math.pow( point_child[a] - point[a], 2d );
                    }
                    distance_child = Math.pow( distance_child, 0.5d );

                    // Keep best
                    if(distance_child < distance_best)
                    {
                        coord_x_best = coord_x_child;
                        coord_y_best = coord_y_child;
                        distance_best = distance_child;
                    }

                    System.out.println("\tx: " + coord_x_child + ", y:" + coord_y_child + "\t:" + format5d.format(distance_child) + "");
                }
            }
            else
            {
                // Not leaf - just all all children to queue
                ArrayList<BoundingBox> items_child = ((BoundingBoxNode) entry.box).Items;
                for(int i = 0; i < items_child.size(); i++)
                {
                    BoundingBox box = items_child.get(i);
                    queue.add( new BoxQueueEntry( box.DistanceToBox(point), box ) );
                }
            }

        }
    }
    //</editor-fold>
}
