"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import  Matern, WhiteKernel, ConstantKernel
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA
MAX_OPTIMIZE_ITERS = 10


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.__x_seen = []
        self.__f_seen = []
        self.__v_seen = []
        sigma_f = 0.15
        nu_f = 2.5
        lenghtscale_f = 1
        sigma_v = 0.0001
        f_kernel = Matern(length_scale=lenghtscale_f, nu=nu_f) + WhiteKernel(noise_level=sigma_f)
        nu_v = 2.5
        lenghtscale_v = 1
        prior_mean_v = 4
        v_kernel = Matern(length_scale=lenghtscale_v, nu=nu_v) + WhiteKernel(noise_level=sigma_v) + ConstantKernel(constant_value=prior_mean_v)
        self.f_model = GaussianProcessRegressor(kernel=f_kernel)
        self.v_model = GaussianProcessRegressor(kernel=v_kernel)
        self.__std_af_fac = 0.02
        self.__std_af_fac_v = 0.001

        self.__expected_contraint_hold_probability = 0.95
        self.__f_max_constraint_holds = None

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        x_next = self.optimize_acquisition_function()
        x_next = np.array(np.clip(x_next, *DOMAIN[0])).reshape(1, 1)
        return x_next

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        # TODO: Implement the acquisition function you want to optimize.

        x = np.atleast_2d(x)
        f_mean, f_std = self.f_model.predict(x, return_std=True)
        v_mean, v_std = self.v_model.predict(x, return_std=True)
        v_margin = self.__get_v_margin(x)
        v_constraint_probab = norm.cdf(v_margin, loc=v_mean, scale=v_std)
        if v_margin < SAFETY_THRESHOLD:
            af_value = (f_mean - self.__f_max_constraint_holds + self.__std_af_fac * f_std + 1) * v_constraint_probab
        else:
            af_value = v_constraint_probab
        return af_value


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        self.__x_seen.append(x)
        self.__f_seen.append(f)
        self.__v_seen.append(v)
        x = np.atleast_2d(x)
        f = np.atleast_2d(f)
        v = np.atleast_2d(v)
        self.f_model.fit(x, f)
        self.v_model.fit(x, v)

        if v > SAFETY_THRESHOLD:
            self.__x_unsafe.append(x)
        
        #add f_max constraint if f is larger than f_max and constraint is satisfied
        f_mean, f_std = self.f_model.predict(x, return_std=True)
        v_margin = self.__get_v_margin(x)
        if v_margin < SAFETY_THRESHOLD:
            if self.__f_max_constraint_holds is None or f_mean > self.__f_max_constraint_holds:
                self.__f_max_constraint_holds = f_mean


    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        for i in range(MAX_OPTIMIZE_ITERS):
            x_opt = self.optimize_acquisition_function()
            x_opt = np.array(np.clip(x_opt, *DOMAIN[0])).reshape(1, 1)
            f_mean, f_std = self.f_model.predict(x_opt, return_std=True)
            v_margin = self.__get_v_margin(x_opt)
            if v_margin < SAFETY_THRESHOLD or i == MAX_OPTIMIZE_ITERS - 1:
                return x_opt
        

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass

    def __get_v_margin(self, x: float):
        """Compute the margin of the constraint at x.

        Parameters
        ----------
        x: float
            x in domain of v

        Returns
        ------
        margin: float
            Margin of the constraint at x
        """
        x = np.atleast_2d(x)
        v_mean, v_std = self.v_model.predict(x, return_std=True)
        v_margin = norm.ppf(self.__expected_contraint_hold_probability, loc=v_mean, scale=v_std)
        return v_margin


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
