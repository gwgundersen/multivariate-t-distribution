from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_almost_equal, assert_equal,
                           assert_array_less, assert_)

from multivariatet import multivariate_t
from scipy.stats import cauchy
from scipy.stats import normaltest
import numpy as np


# These tests were created by running vpa(mvtpdf(...)) in MATLAB. The
# function takes no `mu` parameter. The tests were run as
#
# >> ans = vpa(mvtpdf(x - mu, shape, df));
#
MULTIVARIATE_T_TESTS = [{
    'x': [
        [1, 2],
        [4, 1],
        [2, 1],
        [2, 4],
        [1, 4],
        [4, 1],
        [3, 2],
        [3, 3],
        [4, 4],
        [5, 1],
    ],
    'mu': [0, 0],
    'shape': [[1, 0], [0, 1]],
    'df': 4,
    'ans': [
        0.013972450422333741737457302178882,
        0.0010998721906793330026219646100571,
        0.013972450422333741737457302178882,
        0.00073682844024025606101402363634634,
        0.0010998721906793330026219646100571,
        0.0010998721906793330026219646100571,
        0.0020732579600816823488240725481546,
        0.00095660371505271429414668515889275,
        0.00021831953784896498569831346792114,
        0.00037725616140301147447000396084604
    ]
}, {
    'x': [
        [0.9718, 0.1298, 0.8134],
        [0.4922, 0.5522, 0.7185],
        [0.3010, 0.1491, 0.5008],
        [0.5971, 0.2585, 0.8940],
        [0.5434, 0.5287, 0.9507],
    ],
    'mu': [-1, 1, 50],
    'shape': [
        [1.0000, 0.5000, 0.2500],
        [0.5000, 1.0000, -0.1000],
        [0.2500, -0.1000, 1.0000],
    ],
    'df': 8,
    'ans': [
        0.00000000000000069609279697467772867405511133763,
        0.00000000000000073700739052207366474839369535934,
        0.00000000000000069522909962669171512174435447027,
        0.00000000000000074212293557998314091880208889767,
        0.00000000000000077039675154022118593323030449058,
    ]
}]


def test_pdf_correctness():
    for t in MULTIVARIATE_T_TESTS:
        val = multivariate_t.pdf(t['x'], t['mu'], t['shape'], t['df'])
        assert_array_almost_equal(val, t['ans'])


def test_logpdf_correct():
    for t in MULTIVARIATE_T_TESTS:
        dist = multivariate_t(t['mu'], t['shape'], t['df'])
        # We already test this elsewhere, but let's explicitly confirm the PDF
        # is correct since this test requires checking the log of that value.
        val1 = dist.pdf(t['x'])
        assert_array_almost_equal(val1, t['ans'])
        val2 = dist.logpdf(t['x'])
        assert_array_almost_equal(np.log(val1), val2)


# https://github.com/scipy/scipy/issues/10042#issuecomment-576795195
def test_mvt_with_df_one_is_cauchy():
    x = [9, 7, 4, 1, -3, 9, 0, -3, -1, 3]
    val = multivariate_t.pdf(x, df=1)
    ans = cauchy.pdf(x)
    assert_array_almost_equal(val, ans)


def test_mvt_with_high_df_is_approx_normal():
    # `normaltest` returns the chi-squared statistic and the associated
    # p-value. The null hypothesis is that `x` came from a normal distribution,
    # so a low p-value represents rejecting the null, i.e. that it is unlikely
    # that `x` came a normal distribution.
    P_VAL_MIN = 0.1

    dist = multivariate_t(0, 1, df=100000, seed=1)
    samples = dist.rvs(size=100000)
    _, p = normaltest(samples)
    assert(p > P_VAL_MIN)

    dist = multivariate_t([-2, 3], [[10, -1], [-1, 10]], df=100000, seed=42)
    samples = dist.rvs(size=100000)
    _, p = normaltest(samples)
    assert((p > P_VAL_MIN).all())

    dist = multivariate_t(0, 1, df=np.inf, seed=7)
    samples = dist.rvs(size=100000)
    _, p = normaltest(samples)
    assert(p > P_VAL_MIN)


def test_default_arguments():
    dist = multivariate_t()
    assert_equal(dist.mean, [0])
    assert_equal(dist.shape, [[1]])
    assert(dist.df == 1)


def test_reproducibility():
    rng = np.random.RandomState(4)
    mean = rng.random(3)
    tmp = rng.random((20, 3))
    shape = np.matmul(tmp.T, tmp)
    dist1 = multivariate_t(mean, shape, df=3, seed=2)
    dist2 = multivariate_t(mean, shape, df=3, seed=2)
    samples1 = dist1.rvs(size=10)
    samples2 = dist2.rvs(size=10)
    assert_equal(samples1, samples2)


def test_rvs():
    dist = multivariate_t(mean=range(3))
    samples = dist.rvs(size=1000000)
    print(samples.mean(axis=0))


test_pdf_correctness()
test_logpdf_correct()
test_default_arguments()
test_reproducibility()
test_rvs()
test_mvt_with_df_one_is_cauchy()
test_mvt_with_high_df_is_approx_normal()
