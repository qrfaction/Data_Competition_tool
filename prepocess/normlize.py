


def min_max_scaling(dataset,cols,deal_outlier=False,method=None,upper=None,lower=None):

    def get_bound(data):
        if method == 'default':
            lower_bound = data.min()
            upper_bound = data.max()
        elif method == 'norm_test':
            m = data.mean()
            std = data.std()
            lower_bound = m - 3*std
            upper_bound = m + 3*std
        elif method == 'quantile_test':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            delta = 1.5*(q3 - q1)
            lower_bound = q1 - delta
            upper_bound = q3 + delta
        else:
            assert upper is not None
            assert lower is not None

            lower_bound = lower
            upper_bound = upper
        return lower_bound,upper_bound

    for col in cols:
        lower_bound, upper_bound = get_bound(dataset[col])
        dataset[col] = (dataset[col]-lower_bound)/(upper_bound-lower_bound)
        if deal_outlier:
            dataset[col] = dataset[col].apply(lambda x:1 if x>1 else x)
            dataset[col] = dataset[col].apply(lambda x:0 if x<0 else x)
    return dataset