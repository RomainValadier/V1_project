from __future__ import annotations

import datetime as dt
import importlib.util
import re
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE_PATH = ROOT / "pages" / "8_Nouvel_Individu.py"


def load_page_module():
    fake_streamlit = types.SimpleNamespace(secrets={})
    original_streamlit = sys.modules.get("streamlit")
    sys.modules["streamlit"] = fake_streamlit
    try:
        spec = importlib.util.spec_from_file_location("nouvel_individu_page", PAGE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if original_streamlit is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = original_streamlit


page = load_page_module()


SCHEMA = """
CREATE TABLE individus (
    id TEXT PRIMARY KEY,
    nom TEXT NOT NULL,
    prenom TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    sexe TEXT NOT NULL CHECK (sexe IN ('M','F')),
    date_naissance TEXT,
    role TEXT NOT NULL CHECK (role IN ('athlete','coach','technicien','autre')),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    password_hash TEXT,
    last_login_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE athletes (
    individu_id TEXT PRIMARY KEY REFERENCES individus(id) ON DELETE RESTRICT,
    categorie_poids TEXT,
    taille_cm REAL
);
"""


class NouvelIndividuTests(unittest.TestCase):
    def make_db(self, schema: str = SCHEMA) -> tempfile.NamedTemporaryFile:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        try:
            conn.executescript(schema)
            conn.commit()
        finally:
            conn.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return tmp

    def test_generate_id_format(self):
        ind_id = page.generate_id()

        self.assertTrue(ind_id.startswith("IND"))
        self.assertEqual(19, len(ind_id))
        self.assertRegex(ind_id, re.compile(r"^IND[A-F0-9]{16}$"))
        self.assertNotIn("-", ind_id)

    def test_hash_password(self):
        self.assertIsNone(page.hash_password(""))
        self.assertIsNone(page.hash_password(None))
        self.assertEqual(
            "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b",
            page.hash_password("secret"),
        )

    def test_resolve_db_path_falls_back_without_secrets_file(self):
        original_streamlit = page.st

        class MissingSecrets:
            def get(self, *args, **kwargs):
                raise RuntimeError("No secrets found")

        page.st = types.SimpleNamespace(secrets=MissingSecrets())
        try:
            expected = ROOT / "projet_i.db" if (ROOT / "projet_i.db").exists() else Path("data/projet_I.db")
            self.assertEqual(expected, page.resolve_db_path())
        finally:
            page.st = original_streamlit

    def test_insert_non_athlete_only_creates_individu(self):
        tmp = self.make_db()

        ind_id = page.insert_individu(
            nom="Dupont",
            prenom="Ada",
            email="ada@example.com",
            sexe="F",
            date_naissance=dt.date(1999, 5, 4),
            role="coach",
            is_active=True,
            password="secret",
            categorie_poids=None,
            taille_cm=None,
            db_path=tmp.name,
        )

        conn = sqlite3.connect(tmp.name)
        try:
            individu = conn.execute("SELECT * FROM individus WHERE id = ?", (ind_id,)).fetchone()
            athlete = conn.execute("SELECT * FROM athletes WHERE individu_id = ?", (ind_id,)).fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(individu)
        self.assertIsNone(athlete)
        self.assertTrue(page.email_exists("ada@example.com", tmp.name))

    def test_insert_athlete_normalizes_empty_category(self):
        tmp = self.make_db()

        ind_id = page.insert_individu(
            nom="Martin",
            prenom="Lina",
            email="lina@example.com",
            sexe="F",
            date_naissance=None,
            role="athlete",
            is_active=False,
            password="",
            categorie_poids=page.EMPTY_WEIGHT_OPTION,
            taille_cm=165,
            db_path=tmp.name,
        )

        conn = sqlite3.connect(tmp.name)
        try:
            athlete = conn.execute(
                "SELECT categorie_poids, taille_cm FROM athletes WHERE individu_id = ?",
                (ind_id,),
            ).fetchone()
            individu = conn.execute("SELECT is_active, password_hash FROM individus WHERE id = ?", (ind_id,)).fetchone()
        finally:
            conn.close()

        self.assertIsNone(athlete[0])
        self.assertEqual(165.0, athlete[1])
        self.assertEqual((0, None), individu)

    def test_list_individus_by_role_includes_athlete_profile(self):
        tmp = self.make_db()
        ind_id = page.insert_individu(
            nom="Martin",
            prenom="Lina",
            email="lina.list@example.com",
            sexe="F",
            date_naissance=None,
            role="athlete",
            categorie_poids="−57 kg",
            taille_cm=165,
            db_path=tmp.name,
        )

        rows = page.list_individus_by_role("athlete", tmp.name)

        self.assertEqual(1, len(rows))
        self.assertEqual(ind_id, rows[0]["id"])
        self.assertEqual("−57 kg", rows[0]["categorie_poids"])
        self.assertEqual(165.0, rows[0]["taille_cm"])

    def test_email_exists_can_exclude_current_individu(self):
        tmp = self.make_db()
        ind_id = page.insert_individu(
            nom="Martin",
            prenom="Lina",
            email="lina.exclude@example.com",
            sexe="F",
            date_naissance=None,
            role="athlete",
            db_path=tmp.name,
        )

        self.assertTrue(page.email_exists("lina.exclude@example.com", tmp.name))
        self.assertFalse(page.email_exists("lina.exclude@example.com", tmp.name, exclude_id=ind_id))

    def test_update_athlete_profile(self):
        tmp = self.make_db()
        ind_id = page.insert_individu(
            nom="Martin",
            prenom="Lina",
            email="lina.update@example.com",
            sexe="F",
            date_naissance=None,
            role="athlete",
            categorie_poids="−57 kg",
            taille_cm=165,
            db_path=tmp.name,
        )

        page.update_individu(
            individu_id=ind_id,
            nom="Martin",
            prenom="Line",
            email="line.update@example.com",
            sexe="F",
            date_naissance=dt.date(2001, 1, 2),
            role="athlete",
            categorie_poids="−63 kg",
            taille_cm=168,
            db_path=tmp.name,
        )

        updated = page.get_individu(ind_id, tmp.name)

        self.assertEqual("Line", updated["prenom"])
        self.assertEqual("line.update@example.com", updated["email"])
        self.assertEqual("2001-01-02", updated["date_naissance"])
        self.assertEqual("−63 kg", updated["categorie_poids"])
        self.assertEqual(168.0, updated["taille_cm"])

    def test_update_from_athlete_to_coach_removes_athlete_profile(self):
        tmp = self.make_db()
        ind_id = page.insert_individu(
            nom="Petit",
            prenom="Sam",
            email="sam.switch@example.com",
            sexe="M",
            date_naissance=None,
            role="athlete",
            categorie_poids="−81 kg",
            taille_cm=180,
            db_path=tmp.name,
        )

        page.update_individu(
            individu_id=ind_id,
            nom="Petit",
            prenom="Sam",
            email="sam.switch@example.com",
            sexe="M",
            date_naissance=None,
            role="coach",
            db_path=tmp.name,
        )

        updated = page.get_individu(ind_id, tmp.name)

        self.assertEqual("coach", updated["role"])
        self.assertIsNone(updated["categorie_poids"])
        self.assertIsNone(updated["taille_cm"])

    def test_duplicate_email_raises_integrity_error(self):
        tmp = self.make_db()
        kwargs = {
            "nom": "Durand",
            "prenom": "Noe",
            "email": "noe@example.com",
            "sexe": "M",
            "date_naissance": None,
            "role": "coach",
            "is_active": True,
            "password": None,
            "categorie_poids": None,
            "taille_cm": None,
            "db_path": tmp.name,
        }

        page.insert_individu(**kwargs)

        with self.assertRaises(sqlite3.IntegrityError):
            page.insert_individu(**kwargs)

    def test_failed_athlete_insert_rolls_back_individu(self):
        failing_schema = SCHEMA.replace("taille_cm REAL", "taille_cm REAL CHECK (taille_cm > 190)")
        tmp = self.make_db(failing_schema)

        with self.assertRaises(sqlite3.IntegrityError):
            page.insert_individu(
                nom="Petit",
                prenom="Sam",
                email="sam@example.com",
                sexe="M",
                date_naissance=None,
                role="athlete",
                is_active=True,
                password=None,
                categorie_poids="−81 kg",
                taille_cm=180,
                db_path=tmp.name,
            )

        conn = sqlite3.connect(tmp.name)
        try:
            count = conn.execute("SELECT COUNT(*) FROM individus WHERE email = ?", ("sam@example.com",)).fetchone()[0]
        finally:
            conn.close()

        self.assertEqual(0, count)


if __name__ == "__main__":
    unittest.main()
