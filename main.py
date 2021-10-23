from numpy import array
from numpy.random.mtrand import normal
from utils import run_ransac, plot_fig
from argparse import ArgumentParser
from numpy.random import random

def main(possibility:float,point_count:int,noise_ratio:float,noise_type:str, sample_size:int,goal_inliers_ratio:int, threshold:float, seed:int,save_path:str):
    max_iteration = 1.14514e5
    if noise_type == 'uniform':
        xys = random((point_count, 2)) * 10
        xys[:int(point_count*(1-noise_ratio)), 1:] = xys[:int(point_count*(1-noise_ratio)), :1]
    elif noise_type == 'normal':
        xys = random((point_count, 2)) * 10
        mu = array([normal(loc = 0,scale=1) for _ in range(2)])
        sigma = array([1 for _ in range(2)])
        xys[:,1:]=xys[:,:1]
        for _ in range(int(point_count*noise_ratio)):
            xys[_,]+=normal(mu,sigma)
    else :
        raise NotImplementedError
    m, b = run_ransac(xys, sample_size,possibility, int(
        goal_inliers_ratio*point_count), max_iteration, threshold, seed)
    a, b, c = m
    plot_fig(xys, a, b, c,threshold,save_path)
    return


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--possibility',type=float,default=0.99)
    argparser.add_argument('--noise-ratio',type=float,default =0.5)
    argparser.add_argument('--noise-type',type=str,choices=['normal','uniform'],default='uniform')
    argparser.add_argument('--sample-size', type=int, default=4)
    argparser.add_argument('--goal-inliers-ratio', type=float, default=0.4)
    argparser.add_argument('--point-count', type=int, default=100)
    argparser.add_argument('--threshold', type=float, default=0.3)
    argparser.add_argument('--seed', required=False)
    argparser.add_argument('--save-path',type=str,default='./RANSAC_result.png')
    args = argparser.parse_args()
    main(**vars(args))
