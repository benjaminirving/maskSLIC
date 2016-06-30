# maskSLIC
Simple linear iterative clustering (SLIC) in a region of interest


This code demonstrates the adaption of SLIC for a defined region of interest. 
The main contribution is in the placement of seed points within the ROI. 

The code is a modification of the SLIC implementation provided by Scikit-image (http://scikit-image.org/)

An online demo is available at: http://maskslic.birving.com/index

An outline of the method is available at: [arxiv.org]

Figure 1: An example region and the automatically placed seed points using this method. 

![seed points][outputs/p1.png]

Figure 2: The final superpixel regions within the ROI

![superpixels][outputs/p2.png]


