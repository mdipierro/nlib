## Annotated Algorithms in Python (3.8)

### With applications in Physics, Biology, and Finance

The complete book in PDF is now available under a [Creative Commons BY-NC-ND License](http://creativecommons.org/licenses/by-nc-nd/3.0/legalcode):

[DOWNLOAD BOOK IN PDF](https://raw.githubusercontent.com/mdipierro/nlib/master/docs/book_numerical.pdf)

The book is also available in printed form from Amazon:

[Amazon](http://www.amazon.com/Annotated-Algorithms-Python-Applications-Physics/dp/0991160401)

## The nlib library

The book builds a numerical library from the ground up, called src/nlib.py.
It is a pure python library for numerical computations. It doesn't require numpy.

## Usage

    >>> from nlib import *

## Linear algebra example

    >>> A = Matrix([[1,2],[4,9]])
    >>> print 1/A 
    >>> print (A+2)*A
    >>> B = Matrix(2,2,lambda i,j: i+j**2)

## Fitting

    >>> points = [(x0,y0,dy0), (x1,y1,dy1), (x2,y2,dy2), ...]
    >>> coefficients, chi2, fitting_function = fit_least_squares(points,POLYNOMIAL(2))
    >>> for x,y,dy in points:
    >>>     print x, y, '~', fitting_function(x)

## Solvers
    
    >>> from math import sin
    >>> def f(x): return sin(x)-1+x
    >>> x0 = solve_newton(f, 0.0, ap=0.01, rp=0.01, ns=100)
    >>> print 'f(%s)=%s ~ 0' % (x0, f(x0))

(ap is target absolute precision, rp is target relative precision, ns is max number of steps)

## Optimizers

    >>> def f(x): return (sin(x)-1+x)**2
    >>> x0 = optimize_newton(f, 0.0, ap=0.01, rp=0.01, ns=100)
    >>> print 'f(%s)=%s ~ min f' % (x0, f(x0))    
    >>> print 'f'(%s)=%s ~ 0' % (x0, D(f)(x0))    

## Statistics

    >>> x = [random.random() for k in range(100)]
    >>> print 'mu     =', mean(x)
    >>> print 'sigma  =', sd(x)
    >>> print 'E[x]   =', E(lambda x:x,    x)
    >>> print 'E[x^2] =', E(lambda x:x**2, x)
    >>> print 'E[x^3] =', E(lambda x:x**3, x)
    >>> y = [random.random() for k in range(100)]
    >>> print 'corr(x,y) = ', correlation(x,y)
    >>> print 'cov(x,y)  = ', covariance(x,y)

## Finance

    >>> google = YStock('GOOG')
    >>> current = google.current()
    >>> print current['price']                                                                          
    >>> print current['market_cap']                                                                
    >>> for day in google.historical():
    >>>     print day['date'], day['adjusted_close'], day['log_return']

## Persistant Storage

    >>> d = PersistentDictionary(path='test.sqlite')
    >>> d['key'] = 'value'
    >>> print d['key']
    >>> del d['key']

d works like a drop-in preplacement for any normal Python dictionary except that the data is stored in a sqlite database in a file called  "test.sqlite" so it is still there if you re-start the program. Kind of like the shelve module but shelve files cannot safely be accessed by multiple threads/processes unless locked and locking the entire file is not efficient.

## Neural Network

    >>> pat = [[[0,0], [0]], [[0,1], [1]], [[1,0], [1]], [[1,1], [0]]]
    >>> n = NeuralNetwork(2, 2, 1)
    >>> n.train(pat)
    >>> n.test(pat)
    [0, 0] -> [0.00...]
    [0, 1] -> [0.98...]
    [1, 0] -> [0.98...]
    [1, 1] -> [-0.00...]

## Plotting

    >>> data = [(x0,y0), ...]
    >>> Canvas(title='my plot').plot(data, color='red').save('myplot.png')

nlib plotting requires matplotlib/numpy for the Canvas object only
plots are chainable. methods: .plot, .hist, .errorbar, .ellipses

## Complete list of functions/classes

    CONSTANT
    CUBIC
    Canvas
    Cholesky
    Cluster
    D
    DD
    Dijkstra
    DisjointSets
    E
    Ellipse
    HAVE_MATPLOTLIB
    Jacobi_eigenvalues
    Kruskal
    LINEAR
    MCEngine
    MCG
    Markowitz
    MarsenneTwister
    Matrix
    NeuralNetwork
    POLYNOMIAL
    PersistentDictionary
    Prim
    PrimVertex
    QUADRATIC
    QUARTIC
    QuadratureIntegrator
    RandomSource
    StringIO
    Trader
    YStock
    bootstrap
    breadth_first_search
    compute_correlation
    condition_number
    confidence_intervals
    continuum_knapsack
    correlation
    covariance
    decode_huffman
    depth_first_search
    encode_huffman
    fib
    fit
    fit_least_squares
    gradient
    hessian
    integrate
    integrate_naive
    integrate_quadrature_naive
    invert_bicgstab
    invert_minimum_residual
    is_almost_symmetric
    is_almost_zero
    is_positive_definite
    jacobian
    lcs
    leapfrog
    make_maze
    mean
    memoize
    memoize_persistent
    needleman_wunsch
    norm
    optimize_bisection
    optimize_golden_search
    optimize_newton
    optimize_newton_multi (multi-dimentional optimizer)
    optimize_newton_multi_imporved
    optimize_secant
    partial
    random
    resample
    sd
    solve_bisection
    solve_fixed_point
    solve_newton
    solve_newton_multi (multi-dimensional solver)
    solve_secant
    variance

## License

Created by Massimo Di Pierro (massimo.dipierro@gmail.com) @2016 BSDv3 License
