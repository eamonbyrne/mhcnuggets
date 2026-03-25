"""
Utilities for validating prediction inputs and surfacing clearer runtime
errors without importing the model stack.
"""

import os

try:
    import cPickle as pickle
except:
    import pickle

from mhcnuggets.src.find_closest_mhcI import closest_allele as closest_mhcI
from mhcnuggets.src.find_closest_mhcII import closest_allele as closest_mhcII
from mhcnuggets.src.aa_embeddings import NUM_AAS
from mhcnuggets.src.aa_embeddings import MHCI_MASK_LEN, MHCII_MASK_LEN

MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')


class PredictionError(Exception):
    """
    Raised when a prediction run cannot be completed with a clear
    user-facing explanation.
    """


def get_class_settings(class_):
    """
    Validate the requested MHC class and return model settings.
    """

    class_upper = class_.upper()
    if class_upper == 'I':
        return class_upper, MHCI_MASK_LEN, (MHCI_MASK_LEN, NUM_AAS)
    if class_upper == 'II':
        return class_upper, MHCII_MASK_LEN, (MHCII_MASK_LEN, NUM_AAS)

    raise PredictionError("Invalid MHC class '%s'. Expected 'I' or 'II'." % class_)


def load_peptides(peptides_path):
    """
    Load peptides from a newline-delimited input file.
    """

    if not os.path.isfile(peptides_path):
        raise PredictionError("Peptide input file not found: %s" % peptides_path)

    with open(peptides_path) as peptide_file:
        peptides = [p.strip() for p in peptide_file if p.strip()]

    if not peptides:
        raise PredictionError("Peptide input file is empty or only contains blank lines: %s" % peptides_path)

    return peptides


def validate_output_path(output_path, label):
    """
    Ensure the parent directory for an output file exists.
    """

    if not output_path:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        raise PredictionError("%s directory does not exist: %s" % (label, output_dir))


def resolve_packaged_path(path):
    """
    Resolve paths relative to the installed MHCnuggets package root.
    """

    if os.path.isabs(path):
        return path
    return os.path.join(MHCNUGGETS_HOME, path)


def load_pickle(path, description):
    """
    Load a pickle file with a more readable error if it is missing.
    """

    resolved_path = resolve_packaged_path(path)
    if not os.path.isfile(resolved_path):
        raise PredictionError("%s file not found: %s" % (description, resolved_path))

    with open(resolved_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def resolve_predictor_mhc(class_upper, mhc, pickle_path):
    """
    Resolve the user-requested allele to the concrete model allele.
    """

    try:
        if class_upper == 'I':
            predictor_mhc = closest_mhcI(mhc, pickle_path)
        else:
            predictor_mhc = closest_mhcII(mhc, pickle_path)
    except ValueError as exc:
        raise PredictionError(str(exc))

    if not predictor_mhc:
        raise PredictionError("Could not resolve allele '%s' to a supported Class %s model." %
                              (mhc, class_upper))

    return predictor_mhc


def resolve_model_weights_path(model_weights_path, predictor_mhc, ba_models):
    """
    Resolve the concrete weight file that will be loaded for prediction.
    """

    if model_weights_path != "saves/production/":
        if not os.path.isfile(model_weights_path):
            raise PredictionError("User-specified model weights file not found: %s" % model_weights_path)
        return model_weights_path

    base_path = resolve_packaged_path(model_weights_path)
    ba_path = os.path.join(base_path, predictor_mhc + '_BA.h5')
    ba_to_hlap_path = os.path.join(base_path, predictor_mhc + '_BA_to_HLAp.h5')

    if ba_models:
        if not os.path.isfile(ba_path):
            raise PredictionError("Binding-affinity model weights not found for allele '%s': %s" %
                                  (predictor_mhc, ba_path))
        return ba_path

    if os.path.isfile(ba_to_hlap_path):
        return ba_to_hlap_path

    if os.path.isfile(ba_path):
        return ba_path

    raise PredictionError("No production model weights found for allele '%s'. Checked %s and %s." %
                          (predictor_mhc, ba_to_hlap_path, ba_path))
