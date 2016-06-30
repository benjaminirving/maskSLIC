
#include <cmath>
#include <algorithm>
#include <iostream>

#include "processing.h"

using namespace std;


int main()
{

    // Test vectors
    vector <int> x{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector <int> y{3, 5, 8, 2, 1, 0, 0, 11, 1, 5};
    vector <int> z{1, 2, 4, 5, 0, 1, 12, 3, 4, 5};
    vector <int> dist;

    // Run function
    dist = get_mean_point_distance(x, y, z);

    // Display output
    for (int ii = 0; ii < dist.size(); ii++)
    {
        cout << "Pos: " << ii << " val: " << dist[ii] << endl;
    }

    return 1;
}
