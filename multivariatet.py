"""Multivariate t-distribution.

Author: Gregory Gundersen 2020. Architecture based on SciPy's
`_multivariate.py` module by Joris Vankerschaver 2013.
"""

import numpy as np
from   scipy._lib._util import check_random_state
from   scipy.stats._multivariate import _PSD, multi_rv_generic, multi_rv_frozen
from   scipy.special import gammaln


# -----------------------------------------------------------------------------

class multivariate_t_gen(multi_rv_generic):

    def __init__(self, seed=None):
        """Initialize a multivariate t-distributed random variable.

        Parameters
        ----------
        seed : Random state.
        """
        self._random_state = check_random_state(seed)

    def __call__(self, mean=None, shape=1, df=1, seed=None):
        """Create a frozen multivariate t-distribution. See
        `multivariate_t_frozen` for parameters.
        """
        return multivariate_t_frozen(mean=mean, shape=shape, df=df, seed=seed)

    def pdf(self, x, mean=None, shape=1, df=1):
        """Multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        mean : array_like, optional
            Mean of the distribution (default zero).
        shape : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).
        df : Degrees of freedom.

        Returns
        -------
        logpdf : Probability density function evaluated at `x`.

        Examples
        --------
        FIXME.
        """
        dim, mean, shape, df = self._process_parameters(mean, shape, df)
        x = self._process_quantiles(x, dim)
        shape_info = _PSD(shape)
        lp = self._logpdf(x, mean, shape_info.U, shape_info.log_pdet, df, dim)
        return np.exp(lp)

    def logpdf(self, x, mean=None, shape=1, df=1):
        """Log of the multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        mean : array_like, optional
            Mean of the distribution (default zero).
        shape : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).
        df : Degrees of freedom.

        Returns
        -------
        logpdf : Log of the probability density function evaluated at `x`.

        Examples
        --------
        FIXME.
        """
        dim, mean, shape, df = self._process_parameters(mean, shape, df)
        x = self._process_quantiles(x, dim)
        shape_info = _PSD(shape)
        return self._logpdf(x, mean, shape_info.U, shape_info.log_pdet, df, dim)

    def _logpdf(self, x, mean, U, log_pdet, df, dim):
        """Utility method `pdf`, `logpdf` for parameters.
        """
        dev  = x - mean
        maha = np.square(np.dot(dev, U)).sum(axis=-1)

        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim/2. * np.log(df * np.pi)
        D = 0.5 * log_pdet
        E = -t * np.log(1 + (1./df) * maha)

        return A - B - C - D + E

    def rvs(self, mean=None, shape=1, df=1, size=1, random_state=None):
        """Draw random samples from a multivariate t-distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        mean : array_like, optional
            Mean of the distribution (default zero).
        shape : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).
        df : Degrees of freedom.

        Returns
        -------

        Examples
        --------
        FIXME.
        """
        dim, mean, shape, df = self._process_parameters(mean, shape, df)
        if random_state is not None:
            rng = check_random_state(random_state)
        else:
            rng = self._random_state

        if df == np.inf:
            x = np.ones(size)
        else:
            x = rng.chisquare(df, size=size) / df

        z = rng.multivariate_normal(np.zeros(dim), shape, size=size)
        samples = mean + z / np.sqrt(x)[:, None]
        return samples

    def _process_quantiles(self, x, dim):
        """Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        return x

    def _process_parameters(self, mean, shape, df):
        """Infer dimensionality from mean array and shape matrix, handle
        defaults, and ensure compatible dimensions.
        """
        if mean is None and shape is None:
            shape = np.asarray(1, dtype=float)
            dim = 1
        elif mean is None:
            shape = np.asarray(shape, dtype=float)
            if shape.ndim < 2:
                dim = 1
            else:
                dim = shape.shape[0]
            mean = np.zeros(dim)
        else:
            shape = np.asarray(shape, dtype=float)
            mean = np.asarray(mean, dtype=float)
            dim = mean.size

        # FIXME: Why is this here?
        if dim == 1:
            mean.shape = (1,)
            shape.shape = (1, 1)

        if mean.ndim != 1 or mean.shape[0] != dim:
            raise ValueError("Array 'mean' must be a vector of length %d." %
                             dim)
        if shape.ndim == 0:
            shape = shape * np.eye(dim)
        elif shape.ndim == 1:
            shape = np.diag(shape)
        elif shape.ndim == 2 and shape.shape != (dim, dim):
            rows, cols = shape.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(shape.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'mean' is a vector of length %d.")
                msg = msg % (str(shape.shape), len(mean))
            raise ValueError(msg)
        elif shape.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % shape.ndim)

        # Process degrees of freedom.
        if df is None:
            df = 1
        if not isinstance(df, int) and not np.isinf(df):
            raise ValueError("'df' must be an integer or 'np.inf' but is of "
                             "type %s" % type(df))

        return dim, mean, shape, df


class multivariate_t_frozen(multi_rv_frozen):

    def __init__(self, mean=None, shape=1, df=1, seed=None):
        """
        Create a frozen multivariate normal distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        mean : array_like, optional
            Mean of the distribution (default zero).
        shape : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).
        df : Degrees of freedom.

        Examples
        --------
        FIXME.
        """
        self._dist = multivariate_t_gen(seed)
        dim, mean, shape, df = self._dist._process_parameters(mean, shape, df)
        self.dim, self.mean, self.shape, self.df = dim, mean, shape, df
        self.shape_info = _PSD(shape)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        U = self.shape_info.U
        log_pdet = self.shape_info.log_pdet
        return self._dist._logpdf(x, self.mean, U, log_pdet, self.df, self.dim)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        size : integer, optional
            Number of samples to draw (default 1).
        random_state : np.random.RandomState instance
            RandomState used for drawing the random variates.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.
        """
        return self._dist.rvs(mean=self.mean,
                              shape=self.shape,
                              df=self.df,
                              size=size,
                              random_state=random_state)

# -----------------------------------------------------------------------------

multivariate_t = multivariate_t_gen()
