from __future__ import annotations

import numpy as np

LABEL_2BIS_NONE = "aucun"
LABEL_2BIS_A = "2bis_A"
LABEL_2BIS_B = "2bis_B"

LABEL_OK = "ok"

RATIO_MAX_PHYSIO = 1.25
K_REF = 15
N_OK_MAX = 8


def identifier_zones_suspectes(labels):
    zones = []
    i = 0
    total = len(labels)
    while i < total:
        if labels[i] == LABEL_OK:
            i += 1
            continue
        debut = i
        while i < total and labels[i] != LABEL_OK:
            i += 1
        zones.append((debut, i))
    return zones


def calculer_mediane_ref(rr, labels, debut_run, k_ref):
    indices = []
    j = debut_run - 1
    while j >= 0 and len(indices) < int(k_ref):
        if labels[j] == LABEL_OK and np.isfinite(rr[j]) and rr[j] > 0:
            indices.append(j)
        j -= 1
    if len(indices) < 3:
        return None
    values = np.asarray([rr[idx] for idx in indices], dtype=float)
    return float(np.median(values))


def passe_A(rr, labels_courants, label_2bis, zones, n_ok_max, k_ref, ratio_max):
    total = len(labels_courants)
    for debut, fin in zones:
        mediane_ref = calculer_mediane_ref(rr, labels_courants, debut, k_ref)
        if mediane_ref is None:
            continue
        plafond = mediane_ref * float(ratio_max)
        compteur = 0
        k = fin
        while k < total and compteur < int(n_ok_max):
            if labels_courants[k] != LABEL_OK:
                break
            if rr[k] > plafond and label_2bis[k] == LABEL_2BIS_NONE:
                labels_courants[k] = "long"
                label_2bis[k] = LABEL_2BIS_A
                compteur += 1
                k += 1
                continue
            break
    return labels_courants, label_2bis


def run_etape_2bis(rr, labels, params=None):
    """
    Post-classification Lipponen - passe A uniquement.

    Entrees :
      rr     : array-like de floats, intervalles RR bruts (ms)
      labels : list/array de strings, labels issus de Lipponen

    Sorties :
      labels_modifies : list de strings
      label_2bis      : list de strings ('aucun' / '2bis_A')

    Ne modifie pas les arrays d'entree.
    """
    p = {
        "ratio_max": RATIO_MAX_PHYSIO,
        "k_ref": K_REF,
        "n_ok_max": N_OK_MAX,
    }
    if params:
        p.update(params)

    rr_array = np.asarray(rr, dtype=float)
    labels_courants = list(labels)
    label_2bis = [LABEL_2BIS_NONE] * len(labels_courants)

    zones = identifier_zones_suspectes(labels_courants)
    if not zones:
        return labels_courants, label_2bis

    labels_courants, label_2bis = passe_A(
        rr_array,
        labels_courants,
        label_2bis,
        zones,
        p["n_ok_max"],
        p["k_ref"],
        p["ratio_max"],
    )
    return labels_courants, label_2bis
