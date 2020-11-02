import numpy as np
def snr_akshay(truth, x_in, x_out):
    # Truth (=signals tensor) is of size 1800x1x1024 (or so)
    # x_in (=measurements tensor) is of size 1800x1x1024 (or so)
    # x_out (=preds tensor) is of size 1800x1x1024 (or so)
    EPS = 1e-32
    n_signals = truth.shape[0]
    sig_norms = np.array([np.linalg.norm(truth[i].ravel()) + EPS for i in range(n_signals)]) # np.ravel() Return a contiguous flattened array.
    errs_in = np.array([np.linalg.norm((truth[i] - x_in[i]).ravel()) + EPS for i in range(n_signals)])
    errs_out = np.array([np.linalg.norm((truth[i] - x_out[i]).ravel()) + EPS for i in range(n_signals)])

    snr_in = 20.*(np.log10(sig_norms) - np.log10(errs_in))
    snr_out = 20.*(np.log10(sig_norms) - np.log10(errs_out))
    avg_snr_in = np.mean(snr_in)
    avg_snr_out = np.mean(snr_out)
    avg_snr_gain = avg_snr_out - avg_snr_in
    return avg_snr_in, avg_snr_out, avg_snr_gain

