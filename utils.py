from random import seed, randint
from numpy import ones, abs, mgrid, sqrt
from numpy.linalg import svd
from numpy.random import random
from matplotlib.pyplot import scatter, plot, show, savefig
from numpy import ndarray, array
from scipy.optimize import leastsq
from numpy import log, power


def augment(xys) -> ndarray:
    """
    This function realize the augment operation for data.
    @paraemters:
        xys: 2d points in $n \times 2$ matrix form
    @returns:
        axy:augmented data in $n \times 3$ matrix form where the 1-st dimension is all one
    """
    axy = ones((len(xys), 3))
    axy[:, :2] = xys
    return axy


def estimate(xys) -> tuple:
    """
    This function realize the estimate step in RANSAC.
    @parameters:
        xys: 2d points in $n \ttimes 2$ matrix form
    @returns:
        estimated parameter
    """
    X = array([_[0] for _ in xys])
    Y = array([_[1] for _ in xys])
    p = random(2)

    def error(p, x, y):
        k, b = p
        return sqrt((k*x+b-y)**2)

    p = leastsq(error, p, args=(X, Y))
    m = array([p[0][0], -1, p[0][1]])
    return m


def is_inlier(coeffs, xy, threshold) -> bool:
    """
    This function will judge if a point is a inlier point.
    @parameters:
        coeffes : model parameter in vector form
        xy: a 2d point in array form
        threshold: the treshold distance to distinguish between inlier and outlier. 
    @returns:
        True if the point is inlier
    """
    return sqrt(abs(coeffs.dot(augment([xy]).T))) < threshold


def run_ransac(data: ndarray, sample_size: int, possibility: float, goal_inliers: int, max_iterations: int, threshold: float, random_seed=None):
    """
    This is the main framework for RANSAC algorithm
    @parameters:
        data:2d points in $n \ttimes 2$ matrix form
        sample_size: the size of sample that each time sampled from origin data
        goal_inliers: the number of points that supposed to be inliers in the fincal model
        max_iteration: the number of iteration before stop 
        threshold: the treshold distance to distinguish between inlier and outlier. 
    @returns:
        best_model: the best model's parameter
        best_ic: the best inliers' count of the model
    """
    best_inliers = 0
    best_model = None
    seed(random_seed)
    data = list(data)
    i = 0
    inlier_set = []
    while i < max_iterations:
        i += 1
        ic = 0
        if not len(inlier_set):
            for _ in range(int(sample_size)):
                inlier_set.append(data.pop(randint(0, len(data)-1)))
        m = estimate(inlier_set)
        i_index = []
        for j in range(len(data)):
            if is_inlier(m, data[j], threshold):
                ic += 1
                i_index.append(j)
                inlier_set.append(data[j])
        print('# inliers:', len(inlier_set))
        print('estimate:', m)
        e = 1 - len(inlier_set)/(len(inlier_set)+len(data))
        max_iterations = log(1-possibility)/log(1-(1-e)**sample_size)
        if not ic:
            data += inlier_set
            inlier_set.clear()
        else:
            i_index.reverse()
            for j in i_index:
                data.pop(j)
        if ic > best_inliers:
            best_inliers = len(inlier_set)
            best_model = m
            if best_inliers > goal_inliers:
                break
    print('took iterations:', i+1, 'best model:',
          best_model, 'explains:', best_inliers)
    return best_model, best_inliers


def plot_fig(xys, a, b, c,threshold, save_path) -> None:
    """
    This function is used to draw the result of RANSAC algorithm.
    @parameters:
        xys:2d points in $n \ttimes 2$ matrix form
        a,b,c: model parameters
        save_path: where the experiment picture will be save
    @returns:
        None
    """
    scatter(xys.T[0], xys.T[1])
    plot([0, 10], [-c/b, -(c+10*a)/b], color=(0, 1, 0))
    plot([0, 10], [-(c+threshold/sqrt(a**2+b**2))/b, -(c+10*a+threshold/sqrt(a**2+b**2))/b], color=(0, 1, 0))
    plot([0, 10], [-(c-threshold/sqrt(a**2+b**2))/b, -(c+10*a-threshold/sqrt(a**2+b**2))/b], color=(0, 1, 0))
    
    savefig(save_path)
    show()
    return None
