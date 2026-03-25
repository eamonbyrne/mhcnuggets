'''
Predict IC50s for a batch of peptides
using a trained model

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
from mhcnuggets.src.models import get_predictions, mhcnuggets_lstm
from mhcnuggets.src.dataset import mask_peptides
from mhcnuggets.src.dataset import tensorize_keras, map_proba_to_ic50
from mhcnuggets.src.predict_utils import PredictionError, get_class_settings
from mhcnuggets.src.predict_utils import load_peptides, validate_output_path
from mhcnuggets.src.predict_utils import load_pickle, resolve_predictor_mhc
from mhcnuggets.src.predict_utils import resolve_model_weights_path

try:
    from keras.optimizers import Adam
except:
    from tensorflow.keras.optimizers import Adam
import argparse

import os
import sys
import math
MHCNUGGETS_HOME = os.path.join(os.path.dirname(__file__), '..')


def predict(class_, peptides_path, mhc, pickle_path='data/production/examples_per_allele.pkl',
            model='lstm', model_weights_path="saves/production/", output=None,
            mass_spec=False, ic50_threshold=500, max_ic50=50000, embed_peptides=False,
            binary_preds=False, ba_models=False, rank_output=False,
            hp_ic50s_cI_pickle_path='data/production/mhcI/hp_ic50s_cI.pkl',
            hp_ic50s_positions_cI_pickle_path='data/production/mhcI/hp_ic50s_positions_cI.pkl',
            hp_ic50s_hp_lengths_cI_pickle_path='data/production/mhcI/hp_ic50s_hp_lengths_cI.pkl',
            hp_ic50s_first_percentiles_cI_pickle_path='data/production/mhcI/hp_ic50s_first_percentiles_cI.pkl',
            hp_ic50s_cII_pickle_path='data/production/mhcII/hp_ic50s_cII.pkl',
            hp_ic50s_positions_cII_pickle_path='data/production/mhcII/hp_ic50s_positions_cII.pkl',
            hp_ic50s_hp_lengths_cII_pickle_path='data/production/mhcII/hp_ic50s_hp_lengths_cII.pkl',
            hp_ic50s_first_percentiles_cII_pickle_path='data/production/mhcII/hp_ic50s_first_percentiles_cII.pkl'):
    '''
    Prediction protocol
    '''
    class_upper, mask_len, input_size = get_class_settings(class_)
    peptides = load_peptides(peptides_path)
    validate_output_path(output, 'Output')

    if rank_output and output:
        validate_output_path(output, 'Rank output')

    print('Predicting for %d peptides' % (len(peptides)))

    # apply cut/pad or mask to same length
    normed_peptides, original_peptides = mask_peptides(peptides, max_len=mask_len)
    if not normed_peptides:
        raise PredictionError("No valid peptides remained after length filtering. Class %s supports peptides up to %d residues." %
                              (class_upper, mask_len))

    # get tensorized values for prediction
    peptides_tensor = tensorize_keras(normed_peptides, embed_type='softhot')

    # make model
    print('Building model')
    model = mhcnuggets_lstm(input_size)
    predictor_mhc = resolve_predictor_mhc(class_upper, mhc, pickle_path)
    print("Closest allele found", predictor_mhc)

    weights_path = resolve_model_weights_path(model_weights_path, predictor_mhc, ba_models)
    if model_weights_path != "saves/production/":
        print('Predicting with user-specified model: ' + model_weights_path)
    elif ba_models:
        print('Predicting with only binding affinity trained models')
    elif weights_path.endswith('_BA_to_HLAp.h5'):
        print('BA_to_HLAp model found, predicting with BA_to_HLAp model...')
    else:
        print('No BA_to_HLAp model found, predicting with BA model...')
    try:
        model.load_weights(weights_path)
    except Exception as exc:
        raise PredictionError("Failed to load model weights from %s: %s" % (weights_path, exc))

    if mass_spec:
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    else:
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    # test model
    preds_continuous, preds_binary = get_predictions(peptides_tensor, model, binary_preds, embed_peptides, ic50_threshold, max_ic50)
    ic50s = [map_proba_to_ic50(p[0], max_ic50) for p in preds_continuous]

    rank_filehandle = None
    if (rank_output):
        print("Rank output selected, computing peptide IC50 ranks against human proteome peptides...")
        if class_upper == 'I':
            hp_ic50_pickle = load_pickle(hp_ic50s_cI_pickle_path, 'Class I rank IC50')
            ic50_pos_pickle = load_pickle(hp_ic50s_positions_cI_pickle_path, 'Class I rank position')
            hp_lengths_pickle = load_pickle(hp_ic50s_hp_lengths_cI_pickle_path, 'Class I rank peptide length')
            first_percentiles_pickle = load_pickle(hp_ic50s_first_percentiles_cI_pickle_path,
                                                   'Class I rank percentile')
        else:
            hp_ic50_pickle = load_pickle(hp_ic50s_cII_pickle_path, 'Class II rank IC50')
            ic50_pos_pickle = load_pickle(hp_ic50s_positions_cII_pickle_path, 'Class II rank position')
            hp_lengths_pickle = load_pickle(hp_ic50s_hp_lengths_cII_pickle_path, 'Class II rank peptide length')
            first_percentiles_pickle = load_pickle(hp_ic50s_first_percentiles_cII_pickle_path,
                                                   'Class II rank percentile')

        if predictor_mhc not in first_percentiles_pickle:
            raise PredictionError("Rank output is unavailable for allele '%s' in the selected rank data." %
                                  predictor_mhc)
        ic50_ranks = get_ranks(ic50s,hp_ic50_pickle,hp_lengths_pickle,
                               first_percentiles_pickle,ic50_pos_pickle,predictor_mhc)
        if (output):
            (dir_name, full_file_name) = os.path.split(output)
            if '.' in full_file_name:
                (file_name, extension) = os.path.splitext(full_file_name)
                rank_file_name = os.path.join(dir_name, "{}_ranks.{}".format(file_name, extension))
                rank_filehandle = open(rank_file_name, 'w')
            else:
                rank_filehandle = open(output + '_ranks', 'w')
        else:
            rank_filehandle = sys.stdout

    print("Writing output files...")
    # write out results
    if output:
        filehandle = open(output, 'w')
    else:
        filehandle = sys.stdout

    try:
        print(','.join(('peptide', 'ic50')), file=filehandle)
        for i, peptide in enumerate(original_peptides):
            print(','.join((peptide, str(round(ic50s[i],2)))), file=filehandle)
        if (rank_output):
            print(','.join(('peptide', 'ic50', 'human_proteome_rank')), file=rank_filehandle)
            for i, peptide in enumerate(original_peptides):
                print(','.join((peptide, str(round(ic50s[i],2)), str(round(ic50_ranks[i],4)))), file=rank_filehandle)

    finally:
        if output:
            filehandle.close()
        if rank_filehandle and rank_filehandle is not sys.stdout:
            rank_filehandle.close()



def get_ranks(ic50_list, ic50_pickle, hp_lengths_pickle, first_percentiles_pickle, pos_pickle, mhc):
    """
    Get percentile rank of every ic50 in the given list, when compared to peptides from
    the human proteome.
    """
    rank_list=[]
    first_percentile = first_percentiles_pickle[mhc]
    for ic50 in ic50_list:
        if not math.isnan(ic50):
            if ic50 > first_percentile:
                base_ic50_list = ic50_pickle['downsampled'][mhc]
                closest_ind, exact_match = binary_search(base_ic50_list, 0,
                                                         len(base_ic50_list) - 1,ic50)
                if(exact_match):
                    (first_occ, last_occ) = pos_pickle['downsampled'][mhc][ic50]
                    middle_ind = float(first_occ + last_occ) / 2
                    closest_ind = middle_ind
                percentile=(closest_ind + 1) / float(len(base_ic50_list))
            else:
                base_ic50_list = ic50_pickle['first_percentiles'][mhc]
                hp_length = hp_lengths_pickle[mhc]
                closest_ind, exact_match = binary_search(base_ic50_list, 0,
                                                         len(base_ic50_list) - 1,ic50)
                if(exact_match):
                    (first_occ, last_occ) = pos_pickle['first_percentiles'][mhc][ic50]
                    middle_ind = float(first_occ + last_occ) / 2
                    closest_ind = middle_ind
                percentile=(closest_ind + 1) / float(hp_length)
            rank_list.append(percentile)
        else:
            rank_list.append(float('nan'))
    return rank_list


def binary_search(arr, low, high, x):
    # Check base case
    if high >= low:
        mid = (high + low) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            exact_match = True
            return float(mid), exact_match

            # If element is smaller than mid, then it can only
            # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)

            # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        # Element is not present in the array
        # Record the position to the left of our value of interest
        exact_match = False
        return float(high), exact_match

def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Predict IC50 for a batch of peptides using a trained model'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-m', '--model',
                        type=str, default='lstm',
                        help='Type of MHCnuggets model used to predict' +
                              'options are just lstm for now')

    parser.add_argument('-c', '--class',
                        type=str, required=True,
                        help='MHC class - options are I or II')

    parser.add_argument('-s', '--model_weights_path',
                        type=str, required=False, default='saves/production/',
                        help='Path to which the model weights are saved')

    parser.add_argument('-k', '--pickle_path',
                        type=str, required=False, default='data/production/examples_per_allele.pkl',
                        help='Path to which the pickle file is saved')

    parser.add_argument('-p', '--peptides',
                        type=str, required=True,
                        help='New line separated list of peptides')

    parser.add_argument('-a', '--allele',
                        type=str, required=True,
                        help = 'Allele used for prediction')

    parser.add_argument('-e', '--mass_spec', default=False, type=lambda x: (str(x).lower()== 'true'),
                        help='Train on mass spec data if True, binding affinity data if False')

    parser.add_argument('-o', '--output',
                        type=str, default=None,
                        help='Path to output file, if None, ' +
                              'output is written to stdout')

    parser.add_argument('-l', '--ic50_threshold',
                        type=int, default=500,
                        help='Threshold on ic50 (nM) that separates binder/non-binder')

    parser.add_argument('-x', '--max_ic50',
                        type=int, default=50000,
                        help='Maximum ic50 value')

    parser.add_argument('-q', '--embed_peptides',
                        action='store_true', default=False,
                        help='Embedding of peptides used')

    parser.add_argument('-B', '--binary_predictions',
                        action='store_true', default=False,
                        help='Binary prediction')

    parser.add_argument('-M', '--ba_models',
                        action='store_true', default=False,
                        help='Use binding affinity trained models only instead of mass spec trained models')

    parser.add_argument('-r', '--rank_output', type=lambda x: (str(x).lower()== 'true'),
                        default=False,
                        help='Additionally write output files of predicted peptide ic50 binding ' + \
                        'percentiles compared to human proteome peptides')

    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    try:
        predict(model=opts['model'], class_=opts['class'],
                peptides_path=opts['peptides'],
                model_weights_path=opts['model_weights_path'], pickle_path=opts['pickle_path'],
                mhc=opts['allele'], output=opts['output'],mass_spec=opts['mass_spec'],
                ic50_threshold=opts['ic50_threshold'],
                max_ic50=opts['max_ic50'], embed_peptides= opts['embed_peptides'],
                binary_preds=opts['binary_predictions'],ba_models=opts['ba_models'],
                rank_output=opts['rank_output'])
    except PredictionError as exc:
        print("Prediction failed: %s" % exc, file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
