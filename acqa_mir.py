import numpy as np
from mir_eval.chord import QUALITIES, encode_many, validate

def majmin_quality_only(reference_labels, estimated_labels):
    """Compare chords based only on major-minor quality, ignoring roots."""
    validate(reference_labels, estimated_labels)

    maj_semitones = np.array(QUALITIES["maj"][:8])
    min_semitones = np.array(QUALITIES["min"][:8])

    ref_roots, ref_semitones, _ = encode_many(reference_labels, False)
    est_roots, est_semitones, _ = encode_many(estimated_labels, False)

    eq_quality = np.all(np.equal(ref_semitones[:, :8], est_semitones[:, :8]), axis=1)
    comparison_scores = eq_quality.astype(np.float64)

    is_maj = np.all(np.equal(ref_semitones[:, :8], maj_semitones), axis=1)
    is_min = np.all(np.equal(ref_semitones[:, :8], min_semitones), axis=1)
    is_none = np.logical_and(ref_roots < 0, np.all(ref_semitones == 0, axis=1))

    comparison_scores[(is_maj + is_min + is_none) == 0] = -1

    return comparison_scores


def majmin_inv_quality_only(reference_labels, estimated_labels):
    """
    Compare chords based only on major/minor triad quality and bass note,
    ignoring root comparison. Chords outside maj/min/N are ignored.
    The bass note must still exist in the triad.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        1.0 if triad quality and bass match and inversion valid,
        0.0 if mismatch,
        -1.0 if out of gamut.
    """
    validate(reference_labels, estimated_labels)

    # Triad templates
    maj_semitones = np.array(QUALITIES["maj"][:8])
    min_semitones = np.array(QUALITIES["min"][:8])

    # Encode chords
    ref_roots, ref_semitones, ref_bass = encode_many(reference_labels, False)
    est_roots, est_semitones, est_bass = encode_many(estimated_labels, False)

    # Quality match only (ignoring root)
    eq_quality = np.all(np.equal(ref_semitones[:, :8], est_semitones[:, :8]), axis=1)
    eq_bass = ref_bass == est_bass

    comparison_scores = (eq_quality * eq_bass).astype(np.float64)

    # Check for valid chord types (maj, min, N)
    is_maj = np.all(np.equal(ref_semitones[:, :8], maj_semitones), axis=1)
    is_min = np.all(np.equal(ref_semitones[:, :8], min_semitones), axis=1)
    is_none = np.logical_and(ref_roots < 0, np.all(ref_semitones == 0, axis=1))
    comparison_scores[(is_maj + is_min + is_none) == 0] = -1

    # Bass note must exist in the triad (i.e., valid inversion)
    valid_inversion = np.ones(ref_bass.shape, dtype=bool)
    bass_idx = ref_bass >= 0
    valid_inversion[bass_idx] = ref_semitones[bass_idx, ref_bass[bass_idx]]
    comparison_scores[valid_inversion == 0] = -1

    return comparison_scores


def thirds_quality_only(reference_labels, estimated_labels):
    """
    Compare chords based only on third (minor/major) quality, ignoring roots.

    Parameters
    ----------
    reference_labels : list of str
        Reference chord labels.
    estimated_labels : list of str
        Estimated chord labels.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,)
        1.0 if third quality matches (minor/major), 0.0 if it doesn't, 
        -1.0 if out of gamut (e.g., reference is not maj/min/N).
    """
    validate(reference_labels, estimated_labels)
    
    # Get roots and semitone encodings
    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]
    
    # Compare only the third semitone (index 3)
    eq_thirds = ref_semitones[:, 3] == est_semitones[:, 3]
    comparison_scores = eq_thirds.astype(np.float64)
    
    # Mark out-of-gamut (e.g., X chords) as -1.0
    comparison_scores[np.any(ref_semitones < 0, axis=1)] = -1.0
    
    return comparison_scores


def thirds_inv_quality_only(reference_labels, estimated_labels):
    """
    Compare chords based only on third quality and bass note,
    ignoring root comparison.

    Parameters
    ----------
    reference_labels : list of str
        Reference chord labels.
    estimated_labels : list of str
        Estimated chord labels.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        1.0 if third quality and bass match, 0.0 otherwise,
        -1.0 for invalid reference chords (e.g., 'X').
    """
    validate(reference_labels, estimated_labels)

    # Extract chord encodings
    ref_roots, ref_semitones, ref_bass = encode_many(reference_labels, False)
    est_roots, est_semitones, est_bass = encode_many(estimated_labels, False)

    # Compare third quality (minor vs. major) and bass notes
    eq_third = ref_semitones[:, 3] == est_semitones[:, 3]
    eq_bass = ref_bass == est_bass

    # Only require third and bass to match (ignore root)
    comparison_scores = (eq_third * eq_bass).astype(np.float64)

    # Mask invalid chords
    comparison_scores[np.any(ref_semitones < 0, axis=1)] = -1.0

    return comparison_scores

def triads_quality_only(reference_labels, estimated_labels):
    """
    Compare chords based only on triad quality (up to the 5th), ignoring root.

    Parameters
    ----------
    reference_labels : list of str
        Reference chord labels to score against.
    estimated_labels : list of str
        Estimated chord labels to compare.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        1.0 if triad quality matches (first 8 semitones), 0.0 if not,
        -1.0 if the reference chord is invalid (e.g., 'X').
    """
    validate(reference_labels, estimated_labels)

    # Get root and semitone encodings
    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]

    # Compare triad structure only (first 8 semitones: root → 5th)
    eq_semitones = np.all(np.equal(ref_semitones[:, :8], est_semitones[:, :8]), axis=1)
    comparison_scores = eq_semitones.astype(np.float64)

    # Ignore invalid reference chords (e.g., 'X')
    comparison_scores[np.any(ref_semitones < 0, axis=1)] = -1.0

    return comparison_scores

def triads_inv_quality_only(reference_labels, estimated_labels):
    """
    Compare chords based on triad structure (to 5th) and bass note,
    ignoring the root. Returns -1 for invalid reference chords.

    Parameters
    ----------
    reference_labels : list of str
        Reference chord labels.
    estimated_labels : list of str
        Estimated chord labels.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        1.0 if triad quality and bass match, 0.0 if not, -1.0 if invalid.
    """
    validate(reference_labels, estimated_labels)

    # Extract root, semitone vector, and bass for both ref and est
    ref_roots, ref_semitones, ref_bass = encode_many(reference_labels, False)
    est_roots, est_semitones, est_bass = encode_many(estimated_labels, False)

    # Match on triad structure (first 8 semitones) and bass note only
    eq_semitones = np.all(np.equal(ref_semitones[:, :8], est_semitones[:, :8]), axis=1)
    eq_basses = ref_bass == est_bass
    comparison_scores = (eq_semitones * eq_basses).astype(np.float64)

    # Mask invalid chords (e.g., "X")
    comparison_scores[np.any(ref_semitones < 0, axis=1)] = -1.0

    return comparison_scores

def tetrads_quality_only(reference_labels, estimated_labels):
    """
    Compare chords based on full quality (12D semitone vector),
    ignoring the root.

    Parameters
    ----------
    reference_labels : list of str
        Reference chord labels.
    estimated_labels : list of str
        Estimated chord labels.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        1.0 if full semitone structure matches, 0.0 if not,
        -1.0 if reference chord is invalid.
    """
    validate(reference_labels, estimated_labels)

    # Extract root and full 12-dim semitone vector
    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]

    # Compare full structure (including 7ths, 9ths, etc.)
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    comparison_scores = eq_semitones.astype(np.float64)

    # Ignore 'X' chords (invalid semitone vectors)
    comparison_scores[np.any(ref_semitones < 0, axis=1)] = -1.0

    return comparison_scores

def tetrads_inv_quality_only(reference_labels, estimated_labels):
    """
    Compare chords based on full tetrad quality (12 semitones) and bass note,
    ignoring the root. Returns -1.0 for invalid reference chords (e.g., 'X').

    Parameters
    ----------
    reference_labels : list of str
        Reference chord labels.
    estimated_labels : list of str
        Estimated chord labels.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        1.0 if full semitone structure and bass match,
        0.0 if mismatch,
        -1.0 if reference chord is invalid.
    """
    validate(reference_labels, estimated_labels)

    ref_roots, ref_semitones, ref_bass = encode_many(reference_labels, False)
    est_roots, est_semitones, est_bass = encode_many(estimated_labels, False)

    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    eq_basses = ref_bass == est_bass

    comparison_scores = (eq_semitones * eq_basses).astype(np.float64)

    # Mask invalid reference chords
    comparison_scores[np.any(ref_semitones < 0, axis=1)] = -1.0

    return comparison_scores


def sevenths_quality_only(reference_labels, estimated_labels):
    """
    Compare chords using full semitone structure for common 7th qualities,
    ignoring root comparison. Chords outside [maj, maj7, 7, min, min7, N] are ignored.

    Parameters
    ----------
    reference_labels : list of str
        Reference chord labels.
    estimated_labels : list of str
        Estimated chord labels.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        1.0 if full semitone structure matches, 0.0 if not,
        -1.0 if the reference chord is not in allowed seventh quality set.
    """
    validate(reference_labels, estimated_labels)

    # Valid 7th-type chord templates
    seventh_qualities = ["maj", "min", "maj7", "7", "min7", ""]
    valid_semitones = np.array([QUALITIES[name] for name in seventh_qualities])

    # Encode chords (ignore roots)
    ref_roots, ref_semitones = encode_many(reference_labels, False)[:2]
    est_roots, est_semitones = encode_many(estimated_labels, False)[:2]

    # Match based on full semitone vectors
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    comparison_scores = eq_semitones.astype(np.float64)

    # Filter out chords not in the MIREX-allowed set
    is_valid = np.array([
        np.all(np.equal(ref_semitones, semitones), axis=1)
        for semitones in valid_semitones
    ])
    comparison_scores[np.sum(is_valid, axis=0) == 0] = -1.0

    return comparison_scores


def sevenths_inv_quality_only(reference_labels, estimated_labels):
    """
    Compare chords using full semitone structure and bass note (inversion),
    ignoring root comparison. Only [maj, min, maj7, 7, min7, N] are evaluated.
    Chords must have valid inversion (bass ∈ chord tones).

    Parameters
    ----------
    reference_labels : list of str
        Reference chord labels.
    estimated_labels : list of str
        Estimated chord labels.

    Returns
    -------
    comparison_scores : np.ndarray, shape=(n,), dtype=float
        1.0 if semitone structure and bass match, 0.0 if mismatch,
        -1.0 if out of gamut (invalid quality or invalid inversion).
    """
    validate(reference_labels, estimated_labels)
    
    # Acceptable MIREX 7th qualities
    seventh_qualities = ["maj", "min", "maj7", "7", "min7", ""]
    valid_semitones = np.array([QUALITIES[name] for name in seventh_qualities])

    # Encode
    ref_roots, ref_semitones, ref_basses = encode_many(reference_labels, False)
    est_roots, est_semitones, est_basses = encode_many(estimated_labels, False)

    # Match only semitones and bass (ignore root)
    eq_semitones = np.all(np.equal(ref_semitones, est_semitones), axis=1)
    eq_basses = ref_basses == est_basses
    comparison_scores = (eq_semitones * eq_basses).astype(np.float64)

    # Filter: only allow valid MIREX seventh chords
    is_valid = np.array([
        np.all(np.equal(ref_semitones, semitones), axis=1)
        for semitones in valid_semitones
    ])
    comparison_scores[np.sum(is_valid, axis=0) == 0] = -1

    # Filter: bass must be a chord tone (valid inversion)
    valid_inversion = np.ones(ref_basses.shape, dtype=bool)
    bass_idx = ref_basses >= 0
    valid_inversion[bass_idx] = ref_semitones[bass_idx, ref_basses[bass_idx]]
    comparison_scores[valid_inversion == 0] = -1

    return comparison_scores

