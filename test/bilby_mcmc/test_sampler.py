import os
import shutil
import unittest

import bilby
from bilby.bilby_mcmc.sampler import Bilby_MCMC, BilbyMCMCSampler
from bilby.bilby_mcmc.utils import ConvergenceInputs
from bilby.core.sampler.base_sampler import SamplerError
import numpy as np
import pandas as pd


class TestBilbyMCMCSampler(unittest.TestCase):
    def setUp(self):
        default_kwargs = Bilby_MCMC.default_kwargs
        default_kwargs["target_nsamples"] = 100
        default_kwargs["L1steps"] = 1
        self.convergence_inputs = ConvergenceInputs(
            **{key: default_kwargs[key] for key in ConvergenceInputs._fields}
        )

        self.outdir = "bilby_mcmc_sampler_test"
        if os.path.isdir(self.outdir) is False:
            os.mkdir(self.outdir)

        def model(time, m, c):
            return time * m + c
        injection_parameters = dict(m=0.5, c=0.2)
        sampling_frequency = 10
        time_duration = 10
        time = np.arange(0, time_duration, 1 / sampling_frequency)
        N = len(time)
        sigma = np.random.normal(1, 0.01, N)
        data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)
        likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

        # From hereon, the syntax is exactly equivalent to other bilby examples
        # We make a prior
        priors = dict()
        priors['m'] = bilby.core.prior.Uniform(0, 5, 'm')
        priors['c'] = bilby.core.prior.Uniform(-2, 2, 'c')
        priors = bilby.core.prior.PriorDict(priors)

        search_parameter_keys = ['m', 'c']
        use_ratio = False

        bilby.core.sampler.base_sampler._initialize_global_variables(
            likelihood,
            priors,
            search_parameter_keys,
            use_ratio,
            priors.sample(),
        )

    def tearDown(self):
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)

    def test_None_proposal_cycle(self):
        with self.assertRaises(SamplerError):
            BilbyMCMCSampler(
                convergence_inputs=self.convergence_inputs,
                proposal_cycle=None,
                beta=1,
                Tindex=0,
                Eindex=0,
                use_ratio=False
            )

    def test_default_proposal_cycle(self):
        sampler = BilbyMCMCSampler(
            convergence_inputs=self.convergence_inputs,
            proposal_cycle="default_noNFnoGMnoKD",
            beta=1,
            Tindex=0,
            Eindex=0,
            use_ratio=False
        )

        nsteps = 0
        while sampler.nsamples < 500:
            sampler.step()
            nsteps += 1
        self.assertEqual(sampler.chain.position, nsteps)
        self.assertEqual(sampler.accepted + sampler.rejected, nsteps)
        self.assertTrue(isinstance(sampler.samples, pd.DataFrame))

    def test_initial_sample_outside_prior_is_rejected_before_the_likelihood(self):
        dump = bilby.core.sampler.base_sampler._sampling_convenience_dump
        calls = []
        original = dump.likelihood.log_likelihood

        def counting_log_likelihood(params):
            calls.append(dict(params))
            return original(params)

        dump.likelihood.log_likelihood = counting_log_likelihood
        try:
            with self.assertRaises(SamplerError):
                BilbyMCMCSampler(
                    convergence_inputs=self.convergence_inputs,
                    proposal_cycle="default_noNFnoGMnoKD",
                    beta=1,
                    Tindex=0,
                    Eindex=0,
                    use_ratio=False,
                    initial_sample_dict=dict(m=10.0, c=0.0),
                )
        finally:
            dump.likelihood.log_likelihood = original
        self.assertEqual(calls, [])

    def test_initial_sample_failing_a_constraint_is_rejected_before_the_likelihood(self):
        dump = bilby.core.sampler.base_sampler._sampling_convenience_dump
        priors = bilby.core.prior.PriorDict(dict(
            m=bilby.core.prior.Uniform(0, 5, 'm'),
            c=bilby.core.prior.Uniform(-2, 2, 'c'),
            m_plus_c=bilby.core.prior.Constraint(0, 1, 'm_plus_c'),
        ))

        def conversion(sample):
            sample = dict(sample)
            sample["m_plus_c"] = sample["m"] + sample["c"]
            return sample

        priors.conversion_function = conversion
        calls = []
        original = dump.likelihood.log_likelihood

        def counting_log_likelihood(params):
            calls.append(dict(params))
            return original(params)

        dump.likelihood.log_likelihood = counting_log_likelihood
        bilby.core.sampler.base_sampler._initialize_global_variables(
            dump.likelihood, priors, ['m', 'c'], False, priors.sample(),
        )
        try:
            with self.assertRaises(SamplerError):
                BilbyMCMCSampler(
                    convergence_inputs=self.convergence_inputs,
                    proposal_cycle="default_noNFnoGMnoKD",
                    beta=1,
                    Tindex=0,
                    Eindex=0,
                    use_ratio=False,
                    initial_sample_dict=dict(m=4.0, c=1.0),
                )
            self.assertEqual(calls, [])
            sampler = BilbyMCMCSampler(
                convergence_inputs=self.convergence_inputs,
                proposal_cycle="default_noNFnoGMnoKD",
                beta=1,
                Tindex=0,
                Eindex=0,
                use_ratio=False,
                initial_sample_dict=dict(m=0.5, c=0.2),
            )
        finally:
            dump.likelihood.log_likelihood = original
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.isfinite(sampler.chain.current_sample[bilby.bilby_mcmc.utils.LOGPKEY]))


def test_get_expected_outputs():
    label = "par0"
    outdir = os.path.join("some", "bilby_pipe", "dir")
    filenames, directories = Bilby_MCMC.get_expected_outputs(
        outdir=outdir, label=label
    )
    assert len(filenames) == 1
    assert len(directories) == 0
    assert os.path.join(outdir, f"{label}_resume.pickle") in filenames


if __name__ == "__main__":
    unittest.main()
