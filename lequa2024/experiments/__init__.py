import numpy as np
import pandas as pd
import quapy as qp
from copy import deepcopy
from quapy.protocol import OnLabelledCollectionProtocol
from time import time

def probability_uniformness_ratio_score(prevs, prevs_hat):
    """Compute xi(prevs) / xi(prevs_hat), where xi(*) is the xi_0 jaggedness score of Bunse et al. (2024), which penalizes any deviation from a uniform distribution."""
    assert prevs.shape == prevs_hat.shape, f'wrong shape {prevs.shape} vs. {prevs_hat.shape}'
    def uniformness(p):
        if len(p.shape) == 2:
            return np.sum((p[:,1:] - p[:,:-1])**2, axis=1) / 2
        elif len(p.shape) == 1:
            return np.sum((p[1:] - p[:-1])**2, axis=1) / 2
        else:
            raise ValueError("Not implemented for len(prevs.shape) > 2")
    return uniformness(prevs_hat) / uniformness(prevs)

def mean_probability_uniformness_ratio_score(prevs, prevs_hat):
    """Compute the average of probability_uniformness_ratio_score."""
    return probability_uniformness_ratio_score(prevs, prevs_hat).mean()

def logodds_uniformness_ratio_score(prevs, prevs_hat):
    """Compute |l_prevs| / |l_prevs_hat|, where l_* is the latent vector associated with some prevalence vector and |*| is the squared L2 norm. This score reflects how uniform prevs is, relative to prevs_hat. Values larger than 1 indicate that prevs is less uniform than prevs_hat, values smaller than 1 indicate that prevs is more uniform than prevs_hat."""
    assert prevs.shape == prevs_hat.shape, f'wrong shape {prevs.shape} vs. {prevs_hat.shape}'
    def uniformness(p):
        p = np.maximum(p, 1e-9) # smoothing before log
        ell = np.log(p) # log-odds, i.e. ell = log(p) translated such that ell[0]=0
        if len(ell.shape) == 2:
            ell -= ell[:,0,None] # row-wise translation
            return np.maximum(np.sum(ell * ell, axis=1), 1e-9) # row-wise dot product
        elif len(ell.shape) == 1:
            ell -= ell[0]
            return np.maximum(np.dot(ell, ell), 1e-9)
        else:
            raise ValueError("Not implemented for len(prevs.shape) > 2")
    return uniformness(prevs_hat) / uniformness(prevs)

def mean_logodds_uniformness_ratio_score(prevs, prevs_hat):
    """Compute the average of logodds_uniformness_ratio_score."""
    return logodds_uniformness_ratio_score(prevs, prevs_hat).mean()

def my_from_name(err_name):
    """A variant of quapy.error.from_name that knows our [mean_]{probability,uniformness}_ratio_score."""
    if err_name == "probability_uniformness_ratio_score":
        return probability_uniformness_ratio_score
    elif err_name == "mean_probability_uniformness_ratio_score":
        return mean_probability_uniformness_ratio_score
    elif err_name == "logodds_uniformness_ratio_score":
        return logodds_uniformness_ratio_score
    elif err_name == "mean_logodds_uniformness_ratio_score":
        return mean_logodds_uniformness_ratio_score
    return qp.error.from_name(err_name)

def my_evaluate(
        model,
        protocol,
        error_metric,
        extra_metrics = [],
        aggr_speedup = 'auto',
        verbose = False
        ):
    """A variant of quapy.evaluation.evaluate that computes extra metrics."""
    if isinstance(error_metric, str):
        error_metric = my_from_name(error_metric)
    true_prevs, estim_prevs = qp.evaluation.prediction(
        model,
        protocol,
        aggr_speedup = aggr_speedup,
        verbose = verbose
    )
    scores = [ error_metric(true_prevs, estim_prevs) ]
    for extra_metric in extra_metrics:
        if isinstance(extra_metric, str):
            extra_metric = my_from_name(extra_metric)
        scores.append(extra_metric(true_prevs, estim_prevs))
    return np.array(scores)

class MyGridSearchQ(qp.model_selection.GridSearchQ):
    """A grid search variant that computes extra metrics and stores results in a DataFrame."""
    def __init__(
            self,
            model,
            param_grid,
            protocol,
            error = qp.error.mae,
            extra_metrics = [],
            refit = True,
            timeout = -1,
            n_jobs = None,
            raise_errors = False,
            verbose = False
            ):
        self.error_name = error # circumvent super-constructor's cast to a Callable
        qp.model_selection.GridSearchQ.__init__(
            self,
            model,
            param_grid,
            protocol,
            error,
            refit,
            timeout,
            n_jobs,
            raise_errors,
            verbose,
        )
        self.extra_metrics = extra_metrics

    def _compute_scores_aggregative(self, training):
        # break down the set of hyperparameters into two: classifier-specific, quantifier-specific
        cls_configs, q_configs = group_params(self.param_grid)

        # train all classifiers and get the predictions
        self._training = training
        cls_outs = qp.util.parallel(
            self._prepare_classifier,
            cls_configs,
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs
        )

        # filter out classifier configurations that yielded any error
        success_outs = []
        for (model, predictions, status, took), cls_config in zip(cls_outs, cls_configs):
            if status.success():
                success_outs.append((model, predictions, took, cls_config))
            else:
                self.error_collector.append(status)

        if len(success_outs) == 0:
            raise ValueError('No valid configuration found for the classifier!')

        # explore the quantifier-specific hyperparameters for each valid training configuration
        aggr_configs = [(*out, q_config) for out, q_config in itertools.product(success_outs, q_configs)]
        aggr_outs = qp.util.parallel(
            self._prepare_aggregation,
            aggr_configs,
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs,
            asarray=False,
        )

        return aggr_outs

    def _compute_scores_nonaggregative(self, training):
        configs = qp.model_selection.expand_grid(self.param_grid)
        self._training = training
        scores = qp.util.parallel(
            self._prepare_nonaggr_model,
            configs,
            seed=qp.environ.get('_R_SEED', None),
            n_jobs=self.n_jobs,
            asarray=False,
        )
        return scores

    def _prepare_aggregation(self, args):
        model, predictions, cls_took, cls_params, q_params = args
        model = deepcopy(model)
        params = {**cls_params, **q_params}

        def job(q_params):
            model.set_params(**q_params)
            model.aggregation_fit(predictions, self._training)
            scores = my_evaluate(
                model,
                protocol = self.protocol,
                error_metric = self.error,
                extra_metrics = self.extra_metrics,
            )
            return scores

        scores, status, aggr_took = self._error_handler(job, q_params)
        self._print_status(params, scores[0], status, aggr_took)
        return model, params, scores, status, (cls_took+aggr_took)

    def _prepare_nonaggr_model(self, params):
        model = deepcopy(self.model)

        def job(params):
            model.set_params(**params)
            model.fit(self._training)
            scores = my_evaluate(
                model,
                protocol = self.protocol,
                error_metric = self.error,
                extra_metrics = self.extra_metrics,
            )
            return scores

        scores, status, took = self._error_handler(job, params)
        self._print_status(params, scores[0], status, took)
        return model, params, scores, status, took

    def fit(self, training):
        if self.refit and not isinstance(self.protocol, OnLabelledCollectionProtocol):
            raise RuntimeWarning(
                f'"refit" was requested, but the protocol does not implement '
                f'the {OnLabelledCollectionProtocol.__name__} interface'
            )

        tinit = time()
        self.error_collector = []

        self._sout(f'starting model selection with n_jobs={self.n_jobs}')
        if self._break_down_fit():
            results = self._compute_scores_aggregative(training)
        else:
            results = self._compute_scores_nonaggregative(training)

        df_results = [] # collector for constructing a pd.DataFrame
        self.param_scores_ = {}
        self.extra_scores_ = { extra_metric: {} for extra_metric in self.extra_metrics }
        self.best_score_ = None
        for model, params, scores, status, took in results:
            evaluation_score = scores[0]
            if status.success():
                if self.best_score_ is None or evaluation_score < self.best_score_:
                    self.best_score_ = evaluation_score
                    self.best_params_ = params
                    self.best_model_ = model
                self.param_scores_[str(params)] = evaluation_score
                df_result = {
                    'params': str(params),
                    'status': status.status,
                    self.error_name: evaluation_score,
                    **params,
                }
                for i_extra_metric, extra_metric in enumerate(self.extra_metrics):
                    self.extra_scores_[extra_metric][str(params)] = scores[1+i_extra_metric]
                    df_result[extra_metric] = scores[1+i_extra_metric]
                df_results.append(df_result)
            else:
                self.param_scores_[str(params)] = status.status
                self.error_collector.append(status)
                df_results.append({
                    'params': str(params),
                    'status': status.status,
                    **params,
                })
        self.param_scores_df_ = pd.DataFrame(df_results)

        tend = time()-tinit

        if self.best_score_ is None:
            raise ValueError('no combination of hyperparameters seemed to work')

        self._sout(
            f'optimization finished: best params {self.best_params_}'
            f'(score={self.best_score_:.5f}) [took {tend:.4f}s]'
        )

        no_errors = len(self.error_collector)
        if no_errors>0:
            self._sout(f'warning: {no_errors} errors found')
            for err in self.error_collector:
                self._sout(f'\t{str(err)}')

        if self.refit:
            if isinstance(self.protocol, OnLabelledCollectionProtocol):
                tinit = time()
                self._sout(f'refitting on the whole development set')
                self.best_model_.fit(training + self.protocol.get_labelled_collection())
                tend = time() - tinit
                self.refit_time_ = tend
            else:
                # already checked
                raise RuntimeWarning(f'the model cannot be refit on the whole dataset')

        return self
