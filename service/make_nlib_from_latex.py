"""
this code generated nlib from the latex source of the book. Usage:

    >>> mkdir code
    >>> python service/make_nlib_from_latex.py docs/book_numerical.py
    >>> cp code/nlib.py nlib.py
"""

import sys, os

ALL = """
__all__ = ['CONSTANT', 'CUBIC', 'Canvas', 'Cholesky', 'Cluster', 'D', 'DD', 'Dijkstra', 'DisjointSets', 'E', 'Ellipse', 'Figure', 'FigureCanvasAgg', 'HAVE_MATPLOTLIB', 'Jacobi_eigenvalues', 'Kruskal', 'LINEAR', 'MCEngine', 'MCG', 'Markowitz', 'MarsenneTwister', 'Matrix', 'NeuralNetwork', 'POLYNOMIAL', 'PersistentDictionary', 'Prim', 'PrimVertex', 'QUADRATIC', 'QUARTIC', 'QuadratureIntegrator', 'RandomSource', 'Trader', 'YStock', 'bootstrap', 'breadth_first_search', 'compute_correlation', 'condition_number', 'confidence_intervals', 'continuum_knapsack', 'correlation', 'covariance', 'decode_huffman', 'depth_first_search', 'encode_huffman', 'fib', 'fit', 'fit_least_squares', 'gradient', 'hessian', 'integrate', 'integrate_naive', 'integrate_quadrature_naive', 'invert_bicgstab', 'invert_minimum_residual', 'is_almost_symmetric', 'is_almost_zero', 'is_positive_definite', 'jacobian', 'lcs', 'leapfrog', 'make_maze', 'mean', 'memoize', 'memoize_persistent', 'needleman_wunsch', 'norm', 'optimize_bisection', 'optimize_golden_search', 'optimize_newton', 'optimize_newton_multi', 'optimize_newton_multi_imporved', 'optimize_secant', 'partial', 'resample', 'sd', 'solve_bisection', 'solve_fixed_point', 'solve_newton', 'solve_newton_multi', 'solve_secant', 'variance']

"""

def check_book(filename,path='code'):
    META = '%%% META:FILE:'
    BEGIN = '\\begin{lstlisting}'
    END = '\\end{lstlisting}'
    HEADER = "# Created by Massimo Di Pierro - BSD License\n"

    testfiles = set()
    headers = set()
    test, tests, mode, code = 0,{}, None, ''
    code = ''
    try: os.unlink('%s/*.py' % path)
    except: pass
    open('code/nlib.py','a').write(ALL)
    for line in open(filename,'r'):
        line = line.rstrip()
        if line.startswith(META):
            mode,code = os.path.join(path,line[len(META):].strip()),''            
            if not mode in headers:
                open(mode,'a').write(HEADER)
                headers.add(mode)
        elif line.startswith(BEGIN):
            continue
        elif mode and line.startswith(END):
            testfiles.add(mode)
            tests[mode] = tests.get(mode,'')
            if code.startswith('>>>'):
                code = '    """\n    %s\n    """' \
                    % '\n    '.join(code.split('\n'))
                code = 'def test%.3i():\n%s\n    pass\n' % (test,code)
                tests[mode] += '\n%s\n' % code
                test += 1
            else:
                open(mode,'a').write(code+'\n')
            mode = None
        elif line.strip()==END:
            mode = None
        elif mode:
            code += line+'\n'
    for mode in testfiles:
        open(mode,'a').write(tests[mode])
        open(mode,'a').write("if __name__=='__main__':\n    import os,doctest\n    if not os.path.exists('images'): os.mkdir('images')\n    doctest.testmod(optionflags=doctest.ELLIPSIS)")
    #os.system('rm %s/tests.log' % path)
    #for mode in testfiles:
    #    #os.system('python %s >> %s/tests.log 2>&1' % (mode,path))
    #    os.system('python %s' % mode)

if __name__=='__main__':
    check_book(sys.argv[1])
