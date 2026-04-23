import mne 

def fit_icainfomax(
    X,
    NumberofSources=None,
    ch_types=None,
    n_subgauss=None,
    max_iter=10000,
    l_rate = 1e-3,
    verbose=False,
):
    """
    X : Mixture Signals, X.shape = (NumberofMixtures, NumberofSamples)

    for more information, visit:
    https://mne.tools/stable/generated/mne.preprocessing.ICA.html

    USAGE:
    Y = fit_icainfomax(X = X, NumberofSources = 3)
    IF GROUND TRUTH IS AVAILABLE:
    Y_ = signed_and_permutation_corrected_sources(S.T, Y.T).T
    """
    NumberofMixtures = X.shape[0]
    if NumberofSources is None:
        NumberofSources = NumberofMixtures
    if ch_types is None:
        ch_types = ["eeg"] * NumberofMixtures
    if n_subgauss is None:
        n_subgauss = NumberofSources
    mneinfo = mne.create_info(NumberofMixtures, 2000, ch_types=ch_types)
    mneobj = mne.io.RawArray(X, mneinfo)
    ica = mne.preprocessing.ICA(
        n_components=NumberofSources,
        method="infomax",
        fit_params={"extended": True,
                    "n_subgauss": n_subgauss,
                    "l_rate": l_rate,
                    "max_iter": max_iter},
        random_state=1,
        verbose=verbose,
    )
    ica.fit(mneobj)
    Se = ica.get_sources(mneobj)
    Y = Se.get_data()
    return Y