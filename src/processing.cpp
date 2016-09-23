
#include "processing.h"

long distance_measure(int ax, int ay, int az, int bx, int by, int bz)
{

    int dx, dy, dz;
    long dmeas;

    dx = (bx - ax);
    dy = (by - ay);
    dz = (bz - az);

    dmeas = dx*dx + dy*dy + dz*dz;

    return dmeas;
}

// Get the vector distance (x, y, z) of the mean distance between pairs of closest points
// - For each point, find it's closest neighbour and save the distance
// - Take the mean of all the values to obtain the mean spacing between points
//
//
vector<float> get_mean_point_distance(vector<int> & x, vector<int> & y, vector<int> & z)
{
    vector<float> dist (3, 0);
    // distance pairs
    vector<int> sx;
    vector<int> sy;
    vector<int> sz;

    long dmeas;
    long dist1;
    long nn;


    for (int m=0; m<x.size(); m++)
    {
        dmeas = numeric_limits<long>::max();
        for (int n=0; n<x.size(); n++)
        {
            // Skip comparing point to itself
            if (n==m)
                continue;

            // Calculate the squared distance between each point and every other
            dist1 = distance_measure(x[m], y[m], z[m], x[n], y[n], z[n]);
            if (dist1 < dmeas)
            {
                // If closer then save the indices for that point
                dmeas = dist1;
                nn=n;
            }

        }

        // Save the closest distance for that point
        sx.push_back(abs(x[m] - x[nn]));
        sy.push_back(abs(y[m] - y[nn]));
        sz.push_back(abs(z[m] - z[nn]));
    }

    float d0m = 0;
    float d1m = 0;
    float d2m = 0;
    for (int r=0; r<sx.size(); r++)
    {
        d0m += sx.at(r);
        d1m += sy.at(r);
        d2m += sz.at(r);
    }

    // Mean distance instead

    dist[0] = d0m/sx.size();
    dist[1] = d1m/sy.size();
    dist[2] = d2m/sz.size();

    // Find the maximum distance between two points in the region and use this for the supervoxel extraction
//    dist[0] = *max_element(sx.begin(), sx.end());
//    dist[1] = *max_element(sy.begin(), sy.end());
//    dist[2] = *max_element(sz.begin(), sz.end());

    return dist;
}