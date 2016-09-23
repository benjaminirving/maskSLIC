# maskSLIC
Simple linear iterative clustering (SLIC) in a region of interest

## Outline
This code demonstrates the adaption of SLIC for a defined region of interest. 
The main contribution is in the placement of seed points within the ROI. 

The code is a modification of the SLIC implementation provided by Scikit-image (http://scikit-image.org/)

An online demo is available at: http://maskslic.birving.com/index

An outline of the method is available at: http://arxiv.org/abs/1606.09518

Figure 1: An example region and the automatically placed seed points using this method. 

![seed points](outputs/p1.png)

Figure 2: The final superpixel regions within the ROI

![superpixels](outputs/p2.png)

## Getting started

Install dependencies  
`pip install -r requirements.txt`

build cython code  
`python setup.py build_ext --inplace`

run example  
`python run_example.py`

Using a python `virtualenv` is reccommended on linux to get the latest versions of the dependencies. 


## Development
This code is still a work in progress.