from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import streamlit as st


CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "nouvel_individu.css"

EMPTY_WEIGHT_OPTION = "— non renseignée —"
MALE_WEIGHT_CATEGORIES = ["−60 kg", "−66 kg", "−73 kg", "−81 kg", "−90 kg", "−100 kg", "+100 kg"]
FEMALE_WEIGHT_CATEGORIES = ["−48 kg", "−52 kg", "−57 kg", "−63 kg", "−70 kg", "−78 kg", "+78 kg"]
SEX_LABELS = {"M": "Masculin", "F": "Féminin"}
ROLE_LABELS = {
    "athlete": "Athlète",
    "coach": "Coach",
    "technicien": "Technicien",
    "autre": "Autre",
}
DIRECTORY_ROLES = ("athlete", "coach", "technicien")


def resolve_db_path() -> Path:
    try:
        configured_path = st.secrets.get("db_path", "data/projet_I.db")
    except Exception:
        configured_path = "data/projet_I.db"

    path = Path(configured_path)
    if path.exists() or configured_path != "data/projet_I.db":
        return path

    project_db = Path(__file__).resolve().parents[1] / "projet_i.db"
    if project_db.exists():
        return project_db
    return path


DB_PATH = resolve_db_path()


def get_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def generate_id() -> str:
    return "IND" + uuid.uuid4().hex[:16].upper()


def hash_password(password: str | None) -> str | None:
    if not password:
        return None
    return hashlib.sha256(password.encode()).hexdigest()


def normalize_weight_category(category: str | None) -> str | None:
    if not category or "non renseignée" in category:
        return None
    return category


def normalize_height(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def email_exists(email: str, db_path: str | Path | None = None, exclude_id: str | None = None) -> bool:
    conn = None
    try:
        conn = get_db(db_path)
        if exclude_id:
            row = conn.execute("SELECT 1 FROM individus WHERE email = ? AND id <> ?", (email, exclude_id)).fetchone()
        else:
            row = conn.execute("SELECT 1 FROM individus WHERE email = ?", (email,)).fetchone()
        return row is not None
    finally:
        if conn is not None:
            conn.close()


def list_individus_by_role(role: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    conn = None
    try:
        conn = get_db(db_path)
        rows = conn.execute(
            """
            SELECT
                i.id,
                i.prenom,
                i.nom,
                i.email,
                i.sexe,
                i.date_naissance,
                i.role,
                a.categorie_poids,
                a.taille_cm
            FROM individus i
            LEFT JOIN athletes a ON a.individu_id = i.id
            WHERE i.role = ?
            ORDER BY lower(i.nom), lower(i.prenom)
            """,
            (role,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        if conn is not None:
            conn.close()


def get_individu(individu_id: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    conn = None
    try:
        conn = get_db(db_path)
        row = conn.execute(
            """
            SELECT
                i.id,
                i.prenom,
                i.nom,
                i.email,
                i.sexe,
                i.date_naissance,
                i.role,
                a.categorie_poids,
                a.taille_cm
            FROM individus i
            LEFT JOIN athletes a ON a.individu_id = i.id
            WHERE i.id = ?
            """,
            (individu_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        if conn is not None:
            conn.close()


def utc_now_iso() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def insert_individu(
    *,
    nom: str,
    prenom: str,
    email: str,
    sexe: str,
    date_naissance: dt.date | None,
    role: str,
    is_active: bool = True,
    password: str | None = None,
    categorie_poids: str | None = None,
    taille_cm: float | None = None,
    db_path: str | Path | None = None,
) -> str:
    conn = None
    ind_id = generate_id()
    now = utc_now_iso()
    try:
        conn = get_db(db_path)
        conn.execute(
            """
            INSERT INTO individus (
                id, nom, prenom, email, sexe, date_naissance, role,
                is_active, password_hash, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ind_id,
                nom,
                prenom,
                email,
                sexe,
                date_naissance.isoformat() if date_naissance else None,
                role,
                1 if is_active else 0,
                hash_password(password),
                now,
                now,
            ),
        )

        if role == "athlete":
            conn.execute(
                """
                INSERT INTO athletes (individu_id, categorie_poids, taille_cm)
                VALUES (?, ?, ?)
                """,
                (ind_id, normalize_weight_category(categorie_poids), normalize_height(taille_cm)),
            )

        conn.commit()
        return ind_id
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()


def update_individu(
    *,
    individu_id: str,
    nom: str,
    prenom: str,
    email: str,
    sexe: str,
    date_naissance: dt.date | None,
    role: str,
    categorie_poids: str | None = None,
    taille_cm: float | None = None,
    db_path: str | Path | None = None,
) -> None:
    conn = None
    now = utc_now_iso()
    try:
        conn = get_db(db_path)
        conn.execute(
            """
            UPDATE individus
            SET nom = ?,
                prenom = ?,
                email = ?,
                sexe = ?,
                date_naissance = ?,
                role = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                nom,
                prenom,
                email,
                sexe,
                date_naissance.isoformat() if date_naissance else None,
                role,
                now,
                individu_id,
            ),
        )

        if role == "athlete":
            conn.execute(
                """
                INSERT INTO athletes (individu_id, categorie_poids, taille_cm)
                VALUES (?, ?, ?)
                ON CONFLICT(individu_id) DO UPDATE SET
                    categorie_poids = excluded.categorie_poids,
                    taille_cm = excluded.taille_cm
                """,
                (individu_id, normalize_weight_category(categorie_poids), normalize_height(taille_cm)),
            )
        else:
            conn.execute("DELETE FROM athletes WHERE individu_id = ?", (individu_id,))

        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()


def load_css() -> str:
    if CSS_PATH.exists():
        return CSS_PATH.read_text(encoding="utf-8")
    return ""


def validate_form(prenom: str, nom: str, email: str) -> list[str]:
    errors = []
    if not prenom.strip():
        errors.append("Le prénom est obligatoire.")
    if not nom.strip():
        errors.append("Le nom est obligatoire.")
    if not email.strip() or "@" not in email.strip():
        errors.append("Un email valide est obligatoire.")
    return errors


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def weight_categories_for_sex(sexe: str) -> list[str]:
    return [EMPTY_WEIGHT_OPTION] + (MALE_WEIGHT_CATEGORIES if sexe == "M" else FEMALE_WEIGHT_CATEGORIES)


def format_person_rows(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    formatted = []
    for row in rows:
        item = {
            "Prénom": row["prenom"],
            "Nom": row["nom"],
            "Email": row["email"],
            "Sexe": SEX_LABELS.get(row["sexe"], row["sexe"]),
            "Naissance": row["date_naissance"] or "—",
        }
        if role == "athlete":
            item["Catégorie"] = row["categorie_poids"] or "—"
            item["Taille"] = f"{row['taille_cm']:.0f} cm" if row["taille_cm"] is not None else "—"
        formatted.append(item)
    return formatted


def open_create_form() -> None:
    st.session_state["individual_form_mode"] = "create"
    st.session_state["editing_individual_id"] = None
    st.session_state["show_new_individual_form"] = True


def open_edit_form(individu_id: str) -> None:
    st.session_state["individual_form_mode"] = "edit"
    st.session_state["editing_individual_id"] = individu_id
    st.session_state["show_new_individual_form"] = True


def close_form() -> None:
    st.session_state["show_new_individual_form"] = False
    st.session_state["individual_form_mode"] = "create"
    st.session_state["editing_individual_id"] = None


def render_header() -> None:
    st.markdown(
        """
        <div class="new-individual-header">
            <div class="new-individual-kicker">Registre équipe</div>
            <h1>Individus</h1>
            <p>Consultez les profils enregistrés puis ajoutez un nouvel utilisateur si besoin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_directory(db_path: str | Path | None = None) -> None:
    st.markdown('<div class="section-chip">Annuaire</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Athlètes", "Coachs", "Techniciens"])
    for tab, role in zip(tabs, DIRECTORY_ROLES, strict=True):
        with tab:
            try:
                rows = list_individus_by_role(role, db_path)
            except sqlite3.Error as exc:
                st.warning(f"Impossible de charger la liste : {exc}")
                continue

            if not rows:
                st.info(f"Aucun {ROLE_LABELS[role].lower()} enregistré.")
                continue

            column_spec = [2.1, 2.2, 1.1, 1.2, 1.4, 1.2] if role == "athlete" else [2.1, 2.8, 1.2, 1.4]
            headers = ["Individu", "Email", "Sexe", "Naissance", "Catégorie", "Taille"] if role == "athlete" else ["Individu", "Email", "Sexe", "Naissance"]
            header_cols = st.columns(column_spec)
            for col, header in zip(header_cols, headers, strict=True):
                col.markdown(f'<div class="directory-header">{header}</div>', unsafe_allow_html=True)

            for row in rows:
                row_cols = st.columns(column_spec)
                full_name = f"{row['prenom']} {row['nom']}"
                if row_cols[0].button(full_name, key=f"edit_{row['id']}", help="Modifier cette fiche", use_container_width=True):
                    open_edit_form(row["id"])
                    st.rerun()
                row_cols[1].markdown(f'<div class="directory-cell">{row["email"]}</div>', unsafe_allow_html=True)
                row_cols[2].markdown(f'<div class="directory-cell">{SEX_LABELS.get(row["sexe"], row["sexe"])}</div>', unsafe_allow_html=True)
                row_cols[3].markdown(f'<div class="directory-cell">{row["date_naissance"] or "—"}</div>', unsafe_allow_html=True)
                if role == "athlete":
                    row_cols[4].markdown(f'<div class="directory-cell">{row["categorie_poids"] or "—"}</div>', unsafe_allow_html=True)
                    height = f"{row['taille_cm']:.0f} cm" if row["taille_cm"] is not None else "—"
                    row_cols[5].markdown(f'<div class="directory-cell">{height}</div>', unsafe_allow_html=True)


def render_open_form_button() -> None:
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("+ Nouvel utilisateur", type="primary", use_container_width=True):
            open_create_form()
            st.rerun()
    with col_b:
        if st.session_state.get("show_new_individual_form"):
            if st.button("Fermer le formulaire", use_container_width=False):
                close_form()
                st.rerun()


def render_creation_form() -> None:
    if not st.session_state.get("show_new_individual_form", False):
        return

    mode = st.session_state.get("individual_form_mode", "create")
    editing_id = st.session_state.get("editing_individual_id")
    edited_person = get_individu(editing_id) if mode == "edit" and editing_id else None
    if mode == "edit" and edited_person is None:
        st.warning("Cet individu n'existe plus.")
        close_form()
        return

    section_title = "Modification" if mode == "edit" else "Création"
    submit_label = "Enregistrer les modifications" if mode == "edit" else "Créer l'individu"
    form_context = f"{mode}_{editing_id or 'new'}"

    st.markdown(f'<div class="section-chip">{section_title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-shell">', unsafe_allow_html=True)

    control_cols = st.columns([1, 1, 2])
    with control_cols[0]:
        role_options = ["athlete", "coach", "technicien", "autre"]
        initial_role = edited_person["role"] if edited_person else "athlete"
        role = st.selectbox(
            "Rôle",
            role_options,
            index=role_options.index(initial_role),
            format_func=lambda value: ROLE_LABELS[value],
            key=f"role_{form_context}",
        )
    with control_cols[1]:
        sex_options = ["M", "F"]
        initial_sexe = edited_person["sexe"] if edited_person else "M"
        sexe = st.selectbox(
            "Sexe",
            sex_options,
            index=sex_options.index(initial_sexe),
            format_func=lambda value: SEX_LABELS[value],
            key=f"sexe_{form_context}",
        )

    with st.form("form_nouvel_individu"):
        st.markdown('<div class="form-section-title">Identité</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            prenom = st.text_input("Prénom *", value=edited_person["prenom"] if edited_person else "", key=f"prenom_{form_context}")
        with col_b:
            nom = st.text_input("Nom *", value=edited_person["nom"] if edited_person else "", key=f"nom_{form_context}")

        col_c, col_d = st.columns([2, 1])
        with col_c:
            email = st.text_input("Email *", value=edited_person["email"] if edited_person else "", key=f"email_{form_context}")
        with col_d:
            date_naissance = st.date_input(
                "Date de naissance",
                value=parse_date(edited_person["date_naissance"]) if edited_person else None,
                key=f"date_naissance_{form_context}",
            )

        categorie_poids = None
        taille_cm = None
        if role == "athlete":
            st.markdown('<div class="form-section-title athlete-title">Profil athlète</div>', unsafe_allow_html=True)
            weight_categories = weight_categories_for_sex(sexe)
            athlete_cols = st.columns(2)
            with athlete_cols[0]:
                initial_category = edited_person["categorie_poids"] if edited_person else None
                category_index = weight_categories.index(initial_category) if initial_category in weight_categories else 0
                categorie_poids = st.selectbox("Catégorie de poids", weight_categories, index=category_index, key=f"categorie_poids_{form_context}")
            with athlete_cols[1]:
                taille_value = int(edited_person["taille_cm"]) if edited_person and edited_person["taille_cm"] is not None else None
                taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=230, value=taille_value, key=f"taille_cm_{form_context}")

        submitted = st.form_submit_button(submit_label, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return

    cleaned_prenom = prenom.strip()
    cleaned_nom = nom.strip()
    cleaned_email = email.strip()
    errors = validate_form(cleaned_prenom, cleaned_nom, cleaned_email)

    if cleaned_email and "@" in cleaned_email and email_exists(cleaned_email, exclude_id=editing_id if mode == "edit" else None):
        errors.append("Un individu avec cet email existe déjà.")

    for error in errors:
        st.error(error)

    if errors:
        return

    try:
        if mode == "edit" and editing_id:
            update_individu(
                individu_id=editing_id,
                nom=cleaned_nom,
                prenom=cleaned_prenom,
                email=cleaned_email,
                sexe=sexe,
                date_naissance=date_naissance,
                role=role,
                categorie_poids=categorie_poids if role == "athlete" else None,
                taille_cm=taille_cm if role == "athlete" else None,
            )
            success_message = "Individu modifié."
        else:
            ind_id = insert_individu(
                nom=cleaned_nom,
                prenom=cleaned_prenom,
                email=cleaned_email,
                sexe=sexe,
                date_naissance=date_naissance,
                role=role,
                categorie_poids=categorie_poids if role == "athlete" else None,
                taille_cm=taille_cm if role == "athlete" else None,
            )
            success_message = f"Individu créé — ID : {ind_id}"
    except sqlite3.IntegrityError as exc:
        st.error(str(exc))
        return

    st.session_state["new_individual_success"] = success_message
    close_form()
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Individus", layout="wide", initial_sidebar_state="expanded")
    css = load_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    render_header()

    success_message = st.session_state.pop("new_individual_success", None)
    if success_message:
        st.success(success_message)

    render_directory()
    render_open_form_button()
    render_creation_form()


if __name__ == "__main__":
    main()
