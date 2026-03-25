"""
Utilities for normalizing user-provided allele names and resolving them
against the canonical allele names used by MHCnuggets models.
"""

import re


_HUMAN_GENE_PREFIXES = [
    ('DPA1', 'DPA1'),
    ('DPB1', 'DPB1'),
    ('DQA1', 'DQA1'),
    ('DQB1', 'DQB1'),
    ('DRB1', 'DRB1'),
    ('DRB3', 'DRB3'),
    ('DRB4', 'DRB4'),
    ('DRB5', 'DRB5'),
    ('DRA1', 'DRA'),
    ('DRA', 'DRA'),
    ('DMA1', 'DMA1'),
    ('DMB1', 'DMB1'),
    ('DOA', 'DOA'),
    ('DOB', 'DOB'),
    ('BF', 'BF'),
    ('RT', 'RT'),
    ('A', 'A'),
    ('B', 'B'),
    ('C', 'C'),
    ('E', 'E'),
    ('F', 'F'),
    ('G', 'G'),
]


def _clean_allele_text(mhc):
    """
    Remove whitespace and standardize separators in allele names.
    """

    mhc = mhc.strip()
    mhc = re.sub(r'\s+', '', mhc)
    mhc = mhc.replace('_', '-')
    mhc = mhc.replace('/', '-')
    mhc = mhc.replace('^', '')
    mhc = re.sub(r'-+', '-', mhc)
    return mhc


def _looks_like_human_allele(mhc):
    """
    Check whether an allele resembles a human HLA-style name.
    """

    cleaned = re.sub(r'[^A-Z0-9]', '', mhc.upper())
    if cleaned.startswith('HLA'):
        cleaned = cleaned[3:]

    for prefix, _canonical_prefix in _HUMAN_GENE_PREFIXES:
        if cleaned.startswith(prefix):
            return True
    return False


def _normalize_digit_suffix(canonical_prefix, suffix):
    """
    Normalize a suffix of allele digits into the house style used by
    MHCnuggets, e.g. A0201 -> A02:01 and DPA10103 -> DPA101:03.
    """

    if not suffix:
        return canonical_prefix

    if not suffix.isdigit():
        return canonical_prefix + suffix

    if len(suffix) <= 2:
        return canonical_prefix + suffix

    fields = [suffix[:2]]
    for i in range(2, len(suffix), 2):
        fields.append(suffix[i:i + 2])

    if len(fields[-1]) == 1 and len(fields) > 1:
        fields[-2] += fields[-1]
        fields = fields[:-1]

    return canonical_prefix + fields[0] + ''.join([':' + field for field in fields[1:]])


def _normalize_human_segment(segment):
    """
    Normalize one HLA chain or allele segment.
    """

    cleaned = re.sub(r'[^A-Z0-9]', '', segment.upper())
    for prefix, canonical_prefix in _HUMAN_GENE_PREFIXES:
        if cleaned.startswith(prefix):
            suffix = cleaned[len(prefix):]
            return _normalize_digit_suffix(canonical_prefix, suffix)
    return segment.replace('*', '')


def _normalize_human_allele(mhc):
    """
    Normalize a human allele into the canonical MHCnuggets naming style.
    """

    cleaned = _clean_allele_text(mhc)
    upper = cleaned.upper()

    if upper.startswith('HLA-'):
        remainder = cleaned[4:]
    elif upper.startswith('HLA'):
        remainder = cleaned[3:]
        if remainder.startswith('-'):
            remainder = remainder[1:]
    else:
        remainder = cleaned

    segments = [segment for segment in remainder.split('-') if segment]
    normalized_segments = [_normalize_human_segment(segment) for segment in segments]

    if not normalized_segments:
        return 'HLA-'
    return 'HLA-' + '-'.join(normalized_segments)


def _normalize_mouse_allele(mhc):
    """
    Normalize a mouse H-2 allele into the canonical MHCnuggets naming style.
    """

    cleaned = _clean_allele_text(mhc).replace('*', '')
    upper = cleaned.upper()

    if upper.startswith('H-2-'):
        remainder = cleaned[4:]
    elif upper.startswith('H-2'):
        remainder = cleaned[3:]
        if remainder.startswith('-'):
            remainder = remainder[1:]
    elif upper.startswith('H2-'):
        remainder = cleaned[3:]
    elif upper.startswith('H2'):
        remainder = cleaned[2:]
        if remainder.startswith('-'):
            remainder = remainder[1:]
    else:
        remainder = cleaned

    remainder = remainder.replace('-', '')
    if not remainder:
        return 'H-2-'

    if len(remainder) == 1:
        normalized_segment = remainder.upper()
    else:
        normalized_segment = remainder[:-1].upper() + remainder[-1].lower()

    return 'H-2-' + normalized_segment


def normalize_allele_name(mhc):
    """
    Normalize a user-provided allele name into a canonical-looking form that
    can be used for exact matching and closest-model fallback.
    """

    if mhc is None:
        return ''

    cleaned = _clean_allele_text(str(mhc))
    upper = cleaned.upper()

    if upper.startswith('H2') or upper.startswith('H-2'):
        return _normalize_mouse_allele(cleaned)

    if upper.startswith('HLA') or _looks_like_human_allele(upper):
        return _normalize_human_allele(cleaned)

    return cleaned.replace('*', '')


def allele_aliases(mhc):
    """
    Generate canonical and punctuation-insensitive aliases for allele matching.
    """

    aliases = set()

    for candidate in [mhc, normalize_allele_name(mhc)]:
        if not candidate:
            continue

        cleaned = _clean_allele_text(candidate).replace('*', '')
        upper = cleaned.upper()
        aliases.add(cleaned)
        aliases.add(upper)

        compact = re.sub(r'[^A-Z0-9]', '', upper)
        if compact:
            aliases.add(compact)

        for prefix in ['HLA-', 'HLA', 'H-2-', 'H-2', 'H2-', 'H2']:
            if upper.startswith(prefix):
                trimmed = cleaned[len(prefix):]
                trimmed_upper = upper[len(prefix):]
                aliases.add(trimmed)
                aliases.add(trimmed_upper)
                aliases.add(re.sub(r'[^A-Z0-9]', '', trimmed_upper))

    return set([alias for alias in aliases if alias])


def resolve_allele(mhc, supported_alleles):
    """
    Resolve a user-provided allele name to a supported canonical allele if
    possible. If no exact supported alias exists, return a normalized allele
    string for downstream closest-model logic.
    """

    supported_lookup = set(supported_alleles)
    alias_to_allele = {}

    for allele in supported_alleles:
        for alias in allele_aliases(allele):
            alias_to_allele.setdefault(alias, allele)

    for alias in allele_aliases(mhc):
        if alias in supported_lookup:
            return alias
        if alias in alias_to_allele:
            return alias_to_allele[alias]

    return normalize_allele_name(mhc)
