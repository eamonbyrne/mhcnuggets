import unittest

from mhcnuggets.src.allele_utils import normalize_allele_name, resolve_allele
from mhcnuggets.src.find_closest_mhcI import closest_allele as closest_mhcI
from mhcnuggets.src.find_closest_mhcII import closest_allele as closest_mhcII


PICKLE_PATH = 'data/production/examples_per_allele.pkl'


class AlleleResolutionTests(unittest.TestCase):

    def test_normalize_human_class_i_variant(self):
        self.assertEqual(normalize_allele_name('A*02:01'), 'HLA-A02:01')
        self.assertEqual(normalize_allele_name('HLAA0201'), 'HLA-A02:01')

    def test_normalize_human_class_ii_variants(self):
        self.assertEqual(normalize_allele_name('DRB1*01:01'), 'HLA-DRB101:01')
        self.assertEqual(normalize_allele_name('DRA1*01:01/DRB1*01:01'),
                         'HLA-DRA01:01-DRB101:01')

    def test_normalize_mouse_variant(self):
        self.assertEqual(normalize_allele_name('h2-kb'), 'H-2-Kb')

    def test_resolve_exact_supported_aliases(self):
        supported = [
            'HLA-A02:01',
            'HLA-DRB101:01',
            'HLA-DPA101:03-DPB102:01',
            'H-2-Kb',
        ]

        self.assertEqual(resolve_allele('A0201', supported), 'HLA-A02:01')
        self.assertEqual(resolve_allele('DRB1*01:01', supported), 'HLA-DRB101:01')
        self.assertEqual(resolve_allele('DPA1*01:03/DPB1*02:01', supported),
                         'HLA-DPA101:03-DPB102:01')
        self.assertEqual(resolve_allele('h2kb', supported), 'H-2-Kb')

    def test_closest_mhci_handles_common_aliases(self):
        self.assertEqual(closest_mhcI('A*02:01', PICKLE_PATH), 'HLA-A02:01')
        self.assertEqual(closest_mhcI('h2-kb', PICKLE_PATH), 'H-2-Kb')

    def test_closest_mhcii_handles_common_aliases(self):
        self.assertEqual(closest_mhcII('DRB1*01:01', PICKLE_PATH), 'HLA-DRB101:01')
        self.assertEqual(closest_mhcII('DPA1*01:03/DPB1*02:01', PICKLE_PATH),
                         'HLA-DPA101:03-DPB102:01')
        self.assertEqual(closest_mhcII('DRA1*01:01/DRB1*01:01', PICKLE_PATH),
                         'HLA-DRA01:01-DRB101:01')


if __name__ == '__main__':
    unittest.main()
