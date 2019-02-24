import numpy as np
import matplotlib as plt

# - Distribution Generator code - #
# - Code taken from Microsoft speaker (Netflix guy) - #

class Generator(object):
    
    def __init__(self):
        self._changepoint = -1
    
    def get(self):
        self._changepoint += 1
        return 1.0
    
class DistributionGenerator(Generator):
    """
    A generator which generates values from a single distribution. This may not immediately
    appear useful for change detection, however if we can model our unchanged data stream
    as a distribution then we can test against false positives by running tests against
    a single distribution.
    
    dist1:  A scipy.stats distribution for before changepoint.
    kwargs: The keyword arguments are passed to the distribution.
    """
    
    def __init__(self, dist, **kwargs):
        self._dist = dist
        self._args = kwargs
        
    def get(self):
        return self._dist.rvs(**self._args)

    
class ChangingDistributionGenerator(Generator):
    """
    A generator which takes two distinct distributions and a changepoint and returns
    random variates from the first distribution until it has reached the changepoint
    when it then switches to the next.
    
    dist1:       A scipy.stats distribution for before changepoint.
    kwargs1:     A map specifying loc and scale for dist1.
    dist2:       A scipy.stats distribution for after changepoint.
    kwargs2:     A map specifying loc and scale for dist2.
    changepoint: The number of values to be generated before switching to dist2.
    """
    
    _position = 0
    
    def __init__(self, dist1, kwargs1, dist2, kwargs2, changepoint):
        self._dist1 = dist1
        self._kwargs1 = kwargs1
        self._dist2 = dist2
        self._kwargs2 = kwargs2
        self._changepoint = changepoint
        
    def get(self):
        self._position += 1
        if self._position <= self._changepoint:
            return self._dist1.rvs(**self._kwargs1)
        else:
            return self._dist2.rvs(**self._kwargs2)

        
class DriftGenerator(Generator):
    """
    A generator which takes two distinct distributions and a changepoint and returns
    random variates from the first distribution until it has reached the changepoint
    when it then drifts to the next.
    
    dist1:       A scipy.stats distribution for before changepoint.
    kwargs1:     A map specifying loc and scale for dist1.
    dist2:       A scipy.stats distribution for after changepoint.
    kwargs2:     A map specifying loc and scale for dist2.
    changepoint: The number of values to be generated before switching to dist2.
    steps:       The number of time steps to spend drifting to dist2.
    """
    
    _position = 0
    
    def __init__(self, dist1, kwargs1, dist2, kwargs2, changepoint, steps):
        self._dist1 = dist1
        self._kwargs1 = kwargs1
        self._dist2 = dist2
        self._kwargs2 = kwargs2
        self._changepoint = changepoint
        self._steps = steps
        
        self._change_gradient = np.linspace(0, 1, self._steps)
        
    def get(self):
        self._position += 1
        if self._position < self._changepoint:
            return self._dist1.rvs(**self._kwargs1)
        if self._position >= self._changepoint and self._position < self._changepoint + self._steps:
            beta = self._change_gradient[self._position - self._changepoint - 1]
            return ((1 - beta) * self._dist1.rvs(**self._kwargs1)) + (beta * self._dist2.rvs(**self._kwargs2))
        else:
            return self._dist2.rvs(**self._kwargs2)
        
class DataBackedGenerator(Generator):
    """
    A generator which takes a vector of values and behaves similarly
    to the other generators here. Returns None if values are requested
    past the end of the supplied vector.
    
    vec:         The vector of values for this generator to produce.
    changepoint: The index at which the change occurs.
    """
    
    _idx = 0
    
    def __init__(self, vec, changepoint):
        self._vec = vec
        self._changepoint = changepoint
        
    def get(self):
        if self._idx < len(self._vec):
            self._idx += 1
            return self._vec[self._idx - 1]
        
class DriftingDistributionGenerator(Generator):
    """
    A generator which takes two distinct distributions and a changepoint and returns
    random variates from the first distribution until it has reached the changepoint
    where it linearly drifts to the second distribution.
    
    dist1:       A scipy.stats distribution for before changepoint.
    kwargs1:     A map specifying loc and scale for dist1.
    dist2:       A scipy.stats distribution for after changepoint.
    kwargs2:     A map specifying loc and scale for dist2.
    changepoint: The number of values to be generated before switching to dist2.
    steps:       The number of steps to take in the drift.
    """
    
    _position = 0
    
    def __init__(self, dist1, kwargs1, dist2, kwargs2, changepoint, steps):
        self._dist1 = dist1
        self._kwargs1 = kwargs1
        self._dist2 = dist2
        self._kwargs2 = kwargs2
        self._changepoint = changepoint
        self._steps = steps
        
        self._change_gradient = np.linspace(0, 1, self._steps)
        
    def get(self):
        self._position += 1
        if self._position <= self._changepoint:
            return self._dist1.rvs(**self._kwargs1)
        elif self._position > self._changepoint and self._position <= self._changepoint + self._steps:
            beta = self._change_gradient[self._position - self._changepoint - 1]
            return ((1.0 - beta) * self._dist1.rvs(**self._kwargs1)) + (beta * self._dist2.rvs(**self._kwargs2))
        else:
            return self._dist2.rvs(**self._kwargs2)

# - End generator code - #

