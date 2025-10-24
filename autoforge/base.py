def require_fit(func):
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_is_fitted", False):
            raise ValueError("Model is not trained. Call `fit()` before using this method.")
        return func(self, *args, **kwargs)
    return wrapper


class BaseEstimator:
    def __init__(self):
        self._is_fitted = False

    def _mark_fitted(self):
        self._is_fitted = True
