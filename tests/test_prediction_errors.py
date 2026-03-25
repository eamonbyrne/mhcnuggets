import os
import tempfile
import unittest

from mhcnuggets.src.find_closest_mhcI import closest_allele as closest_mhcI
from mhcnuggets.src.find_closest_mhcII import closest_allele as closest_mhcII
from mhcnuggets.src.predict_utils import PredictionError, get_class_settings
from mhcnuggets.src.predict_utils import load_peptides, resolve_predictor_mhc


PICKLE_PATH = 'data/production/examples_per_allele.pkl'


class PredictionErrorTests(unittest.TestCase):

    def test_invalid_class_gets_clear_error(self):
        with self.assertRaises(PredictionError) as err:
            get_class_settings('III')
        self.assertIn("Invalid MHC class 'III'", str(err.exception))

    def test_missing_peptide_file_gets_clear_error(self):
        with self.assertRaises(PredictionError) as err:
            load_peptides('/definitely/missing/peptides.txt')
        self.assertIn('Peptide input file not found', str(err.exception))

    def test_blank_peptide_file_gets_clear_error(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as peptide_file:
            peptide_file.write('\n \n')
            peptide_path = peptide_file.name

        try:
            with self.assertRaises(PredictionError) as err:
                load_peptides(peptide_path)
            self.assertIn('empty or only contains blank lines', str(err.exception))
        finally:
            os.unlink(peptide_path)

    def test_invalid_class_i_allele_gets_clear_error(self):
        with self.assertRaises(ValueError) as err:
            closest_mhcI('not-an-allele', PICKLE_PATH)
        self.assertIn('Unsupported Class I allele', str(err.exception))

    def test_invalid_class_ii_allele_gets_clear_error(self):
        with self.assertRaises(ValueError) as err:
            closest_mhcII('not-an-allele', PICKLE_PATH)
        self.assertIn('Unsupported Class II allele', str(err.exception))

    def test_predictor_resolution_wraps_allele_errors(self):
        with self.assertRaises(PredictionError) as err:
            resolve_predictor_mhc('I', 'not-an-allele', PICKLE_PATH)
        self.assertIn('Unsupported Class I allele', str(err.exception))


if __name__ == '__main__':
    unittest.main()
