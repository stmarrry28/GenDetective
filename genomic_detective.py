"""
Геномный детектив: приоритизация генетических вариантов для поиска причины
редкого наследственного заболевания (хакатон СИНТЕЗ, кейс 3, ПИШ ИТМО).

Идея в двух словах:
  1. Берём варианты пациента + их разметку из VEP (ген, тип последствия, частота, оценки вредности).
  2. Отбрасываем частые варианты и безобидные (между генами, "молчащие" и т.п.).
  3. Для каждого оставшегося гена смотрим в справочник HPO: какие симптомы бывают у болезней этого гена.
  4. Сравниваем с симптомами пациента (и учитываем исключённые симптомы).
  5. Проверяем, подходит ли тип наследования (рецессивный/доминантный) к найденным вариантам.
  6. Складываем всё в итоговый балл, сортируем и объясняем.

Результат предназначен только для исследований и обучения и не является клиническим диагнозом.
"""
import glob
import gzip
import json
import math
import os
import re
import urllib.request
from collections import defaultdict

import pandas as pd

# ----------------------------------------------------------------------------
# Настройки (их можно менять из ноутбука)
# ----------------------------------------------------------------------------
DATA_DIR = "data"          # где лежат файлы случаев и справочники HPO
OUT_DIR = "results"        # куда сохраняются таблицы и отчёты

URLS = {
    "genes_to_phenotype.txt": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/genes_to_phenotype.txt",
    "hp.obo": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo",
    "phenotype.hpoa": "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa",
    "clinvar.vcf.gz": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
}

# Веса итогового балла (их подбирают на открытых случаях)
W_VARIANT = 0.35           # вклад "насколько вариант похож на причину"
W_PHENO = 0.65             # вклад "насколько ген подходит к симптомам"
MIN_VARIANT_SCORE = 0.25   # варианты слабее этого вообще не считаем кандидатами
CONFIDENT_SCORE = 0.50     # итоговый балл для "убедительного" кандидата
POSSIBLE_SCORE = 0.35      # ниже этого - "убедительного кандидата нет"
CONFIDENT_MARGIN = 0.08    # отрыв первого места от второго (разных генов)
MIN_SYMPTOMS_FOR_CONFIDENCE = 4  # при меньшем числе симптомов уверенный вывод не делаем

ROOT_TERMS = {"HP:0000001", "HP:0000118", "HP:0000005"}  # слишком общие корни HPO

# ----------------------------------------------------------------------------
# Скачивание и поиск файлов
# ----------------------------------------------------------------------------

def find_file(patterns, folder=None):
    """Ищет первый файл, подходящий под один из шаблонов, во всех подпапках."""
    folder = folder or DATA_DIR
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(folder, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None


def ensure_file(name, required=True):
    """Находит справочник в папке данных или скачивает его."""
    path = find_file([name])
    if path:
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    try:
        print(f"Скачиваю {name} ... (это разовое действие)")
        urllib.request.urlretrieve(URLS[name], path)
        return path
    except Exception as e:  # noqa
        if required:
            raise RuntimeError(
                f"Не получилось скачать {name}: {e}. Скачайте файл вручную и положите в папку {DATA_DIR}."
            )
        print(f"Не удалось скачать {name} ({e}). Продолжаю без него.")
        return None


# ----------------------------------------------------------------------------
# Онтология HPO (словарь симптомов и их "родственных связей")
# ----------------------------------------------------------------------------
class HPO:
    def __init__(self, obo_path):
        self.names, self.parents, self.alt = {}, defaultdict(set), {}
        cur, obsolete = None, False
        with open(obo_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "[Term]":
                    cur, obsolete = None, False
                elif line.startswith("id: HP:"):
                    cur = line[4:]
                elif cur and line.startswith("name: "):
                    self.names[cur] = line[6:]
                elif cur and line.startswith("alt_id: "):
                    self.alt[line[8:]] = cur
                elif cur and line.startswith("is_a: "):
                    self.parents[cur].add(line[6:].split()[0])
                elif cur and line.startswith("replaced_by: "):
                    self.alt[cur] = line[13:]
        self._anc = {}
        self.inherit_terms = {t for t in self.names if "HP:0000005" in self.ancestors(t)}

    def canon(self, t):
        return self.alt.get(t, t)

    def ancestors(self, t):
        """Сам термин + все его "родители" вверх по дереву."""
        if t in self._anc:
            return self._anc[t]
        res = {t}
        for p in self.parents.get(t, ()):
            res |= self.ancestors(p)
        self._anc[t] = res
        return res

    def name(self, t):
        return self.names.get(t, t)

    def search(self, text, limit=15):
        text = text.lower()
        return [(t, n) for t, n in self.names.items() if text in n.lower()][:limit]


def search_hpo(text, limit=15):
    """Поиск кода симптома по названию (по-английски): search_hpo('arachnodactyly')."""
    res = load_resources()["hpo"].search(text, limit)
    for t, n in res:
        print(t, "-", n)
    return res


FREQ_WEIGHT = {
    "HP:0040280": 1.0, "HP:0040281": 0.9, "HP:0040282": 0.6,
    "HP:0040283": 0.3, "HP:0040284": 0.1, "HP:0040285": 0.0,
}


def freq_to_weight(f):
    f = str(f)
    if f in FREQ_WEIGHT:
        return FREQ_WEIGHT[f]
    m = re.match(r"^(\d+)/(\d+)$", f)
    if m and int(m.group(2)) > 0:
        return int(m.group(1)) / int(m.group(2))
    m = re.match(r"^([\d.]+)%$", f)
    if m:
        return float(m.group(1)) / 100
    return 0.5  # частота неизвестна


_RES = {}


def load_resources():
    """Загружает справочники один раз и готовит таблицы 'ген -> болезнь -> симптомы'."""
    if _RES:
        return _RES
    hpo = HPO(ensure_file("hp.obo"))
    g2p = pd.read_csv(ensure_file("genes_to_phenotype.txt"), sep="\t", dtype=str)
    g2p["hpo_id"] = g2p["hpo_id"].map(hpo.canon)
    g2p["w"] = g2p["frequency"].map(freq_to_weight)
    g2p = g2p[g2p["w"] > 0]  # строки, где симптом явно НЕ бывает у болезни (0%), выбрасываем

    # симптомы каждой пары (ген, болезнь): {term: weight}; режим наследования отдельно
    pair_terms = defaultdict(dict)
    pair_inh = defaultdict(set)
    for gene, dis, t, w in zip(g2p.gene_symbol, g2p.disease_id, g2p.hpo_id, g2p.w):
        key = (gene, dis)
        if t in hpo.inherit_terms:
            anc = hpo.ancestors(t)
            if "HP:0000007" in anc:
                pair_inh[key].add("AR")
            if "HP:0000006" in anc:
                pair_inh[key].add("AD")
            if "HP:0001417" in anc:
                pair_inh[key].add("XL")
        elif t not in ROOT_TERMS:
            pair_terms[key][t] = max(w, pair_terms[key].get(t, 0))

    # информационная ценность (IC): редкие симптомы весят больше, чем частые
    dis_terms = defaultdict(set)
    for (gene, dis), terms in pair_terms.items():
        dis_terms[dis] |= set(terms)
    count = defaultdict(int)
    for dis, terms in dis_terms.items():
        exp = set()
        for t in terms:
            exp |= hpo.ancestors(t)
        for t in exp:
            count[t] += 1
    n_dis = max(len(dis_terms), 1)

    def ic(t):
        return -math.log((count.get(t, 0) + 1) / (n_dis + 1))

    # названия болезней
    dnames = {}
    hpoa = ensure_file("phenotype.hpoa", required=False)
    if hpoa:
        d = pd.read_csv(hpoa, sep="\t", comment="#", usecols=["database_id", "disease_name"], dtype=str)
        dnames = dict(d.drop_duplicates("database_id").values)

    gene_pairs = defaultdict(list)
    for (gene, dis) in pair_terms:
        gene_pairs[gene].append(dis)

    _RES.update(hpo=hpo, pair_terms=pair_terms, pair_inh=pair_inh, ic=ic,
                dnames=dnames, gene_pairs=gene_pairs)
    return _RES


# ----------------------------------------------------------------------------
# Чтение данных случая
# ----------------------------------------------------------------------------

def load_phenotype(case_id):
    path = find_file([f"{case_id}_phenotype.json"])
    if not path:
        raise FileNotFoundError(f"Не найден {case_id}_phenotype.json в папке {DATA_DIR}")
    js = json.load(open(path, encoding="utf-8"))
    present, excluded, labels = [], [], {}
    for f in js.get("phenotypicFeatures", []):
        t = f["type"]["id"]
        labels[t] = f["type"].get("label", t)
        (excluded if f.get("excluded") else present).append(t)
    return {"sex": js.get("sex", "UNKNOWN"), "present": present, "excluded": excluded, "labels": labels}


def load_genotypes(case_id):
    """Генотипы вариантов: из таблицы *_variants.tsv, а если её нет - из VCF."""
    p = find_file([f"{case_id}_variants.tsv"])
    if p:
        df = pd.read_csv(p, sep="\t", dtype=str)
        df = df.rename(columns={"genotype": "gtype"})
        df = df[["chrom", "pos", "ref", "alt", "gtype"]]
    else:
        p = find_file([f"{case_id}.vcf", f"{case_id}.vcf.gz"])
        if not p:
            raise FileNotFoundError(f"Не найден ни {case_id}_variants.tsv, ни {case_id}.vcf")
        rows = []
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                c = line.rstrip("\n").split("\t")
                rows.append((c[0], c[1], c[3], c[4], c[9].split(":")[0]))
        df = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt", "gtype"])
    df["chrom"] = df["chrom"].str.replace("chr", "", regex=False)
    df["pos"] = df["pos"].astype(int)
    return df


def load_vep(case_id):
    p = find_file([f"{case_id}_*VEP*", f"{case_id}_vep*", f"{case_id}*vep*"])
    if not p:
        raise FileNotFoundError(
            f"Не найден результат VEP для {case_id} (имя файла должно начинаться с {case_id}_ и содержать VEP)")
    df = pd.read_csv(p, sep="\t", comment=None, dtype=str, na_values=["-", ""], low_memory=False)
    df.columns = [c.lstrip("#") for c in df.columns]
    loc = df["Location"].str.split(":", n=1, expand=True)
    df["chrom"] = loc[0].str.replace("chr", "", regex=False)
    df["pos"] = loc[1].str.split("-").str[0].astype(int)
    df["alt"] = df["Allele"]
    for c in ["CADD_PHRED", "REVEL", "LOEUF", "gnomAD_exomes_AF", "gnomAD_genomes_AF"] + \
            [c for c in df.columns if c.startswith("SpliceAI_pred_DS_")]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ----------------------------------------------------------------------------
# ClinVar (необязательно, но полезно)
# ----------------------------------------------------------------------------
CLINVAR_CACHE = {}
_CLINVAR_FAILED = False


def clinvar_lookup(targets, path=None):
    """targets: набор (chrom, pos, ref, alt). Возвращает {ключ: (значимость, статус, болезнь)} или None."""
    todo = {t for t in targets if t not in CLINVAR_CACHE}
    if todo:
        global _CLINVAR_FAILED
        if _CLINVAR_FAILED and not path:
            return {}
        path = path or ensure_file("clinvar.vcf.gz", required=False)
        if not path:
            _CLINVAR_FAILED = True
            return {}
        positions = {(t[0], t[1]) for t in todo}
        found = {}
        with gzip.open(path, "rt") as f:
            for line in f:
                if line[0] == "#":
                    continue
                c = line.split("\t", 8)
                try:
                    key = (c[0], int(c[1]))
                except ValueError:
                    continue
                if key not in positions:
                    continue
                k4 = (c[0], int(c[1]), c[3], c[4])
                if k4 in todo:
                    info = c[7].rstrip("\n")
                    sig = re.search(r"CLNSIG=([^;]+)", info)
                    rev = re.search(r"CLNREVSTAT=([^;]+)", info)
                    dn = re.search(r"CLNDN=([^;]+)", info)
                    found[k4] = (sig.group(1) if sig else None, rev.group(1) if rev else None,
                                 dn.group(1).replace("_", " ") if dn else None)
        for t in todo:
            CLINVAR_CACHE[t] = found.get(t)
    return {t: CLINVAR_CACHE.get(t) for t in targets}


# ----------------------------------------------------------------------------
# Оценка вариантов
# ----------------------------------------------------------------------------
CONS_SCORE = {
    "transcript_ablation": 1.0, "splice_acceptor_variant": 1.0, "splice_donor_variant": 1.0,
    "stop_gained": 1.0, "frameshift_variant": 1.0, "stop_lost": 0.9, "start_lost": 0.9,
    "inframe_insertion": 0.5, "inframe_deletion": 0.5, "missense_variant": 0.5,
    "protein_altering_variant": 0.5, "splice_region_variant": 0.25,
    "splice_donor_5th_base_variant": 0.25, "splice_donor_region_variant": 0.25,
    "splice_polypyrimidine_tract_variant": 0.25, "start_retained_variant": 0.1,
    "stop_retained_variant": 0.1, "synonymous_variant": 0.05,
}
CONS_RU = {
    "splice_donor_variant": "нарушение сайта сплайсинга (донор)",
    "splice_acceptor_variant": "нарушение сайта сплайсинга (акцептор)",
    "stop_gained": "преждевременный стоп-кодон",
    "frameshift_variant": "сдвиг рамки считывания",
    "missense_variant": "замена аминокислоты",
    "start_lost": "потеря стартового кодона",
    "stop_lost": "потеря стоп-кодона",
    "inframe_deletion": "делеция без сдвига рамки",
    "inframe_insertion": "вставка без сдвига рамки",
    "splice_region_variant": "рядом с сайтом сплайсинга",
}


def rarity_factor(af):
    if af is None or (isinstance(af, float) and math.isnan(af)):
        return 1.0
    if af >= 0.01:
        return 0.0
    if af >= 0.001:
        return 0.3
    if af >= 1e-4:
        return 0.7
    if af >= 1e-5:
        return 0.9
    return 1.0


def score_rows(rows):
    """Оценка одного варианта в одном гене по всем его строкам VEP. Возвращает словарь."""
    best_cons, best_row, best_val = None, None, -1
    for _, r in rows.iterrows():
        terms = str(r["Consequence"]).split("&")
        val = max((CONS_SCORE.get(t, 0.0) for t in terms), default=0.0)
        if r.get("BIOTYPE") != "protein_coding" or "NMD_transcript_variant" in terms \
                or "non_coding_transcript_variant" in terms:
            val *= 0.3
        if val > best_val:
            best_val, best_row = val, r
    r = best_row
    terms = str(r["Consequence"]).split("&")
    main_term = max(terms, key=lambda t: CONS_SCORE.get(t, 0.0))
    impact = best_val

    revel = rows["REVEL"].max() if "REVEL" in rows else float("nan")
    cadd = rows["CADD_PHRED"].max()
    if main_term == "missense_variant" and best_val >= 0.5:
        if not math.isnan(revel):
            impact = 0.8 if revel >= 0.75 else 0.65 if revel >= 0.5 else 0.45 if revel >= 0.25 else 0.3
        elif not math.isnan(cadd):
            impact = 0.65 if cadd >= 25 else 0.5 if cadd >= 20 else 0.25
    sp_cols = [c for c in rows.columns if c.startswith("SpliceAI_pred_DS_")]
    spliceai = rows[sp_cols].max().max() if sp_cols else float("nan")
    if not math.isnan(spliceai):
        s = 0.9 if spliceai >= 0.8 else 0.7 if spliceai >= 0.5 else 0.35 if spliceai >= 0.2 else 0
        impact = max(impact, s)
    if not math.isnan(cadd):
        impact *= 0.85 + 0.15 * min(cadd, 35) / 35

    af = max([x for x in (rows["gnomAD_exomes_AF"].max(), rows["gnomAD_genomes_AF"].max()) if not math.isnan(x)],
             default=float("nan"))
    rar = rarity_factor(af)
    return {
        "impact": impact, "rarity": rar, "score": impact * rar, "consequence": main_term,
        "af": af, "cadd": cadd, "revel": revel, "spliceai": spliceai,
        "hgvsc": (str(r.get("HGVSc")).split(":")[-1] if pd.notna(r.get("HGVSc")) else ""),
        "hgvsp": (str(r.get("HGVSp")).split(":")[-1] if pd.notna(r.get("HGVSp")) else ""),
        "loeuf": rows["LOEUF"].min() if "LOEUF" in rows else float("nan"),
    }


def build_variant_table(vep, geno, use_clinvar=True, clinvar_path=None):
    """Одна строка = один вариант в одном гене, с оценкой."""
    vep = vep[vep["SYMBOL"].notna()]
    merged = vep.merge(geno, on=["chrom", "pos", "alt"], how="inner")
    out = []
    for (chrom, pos, alt, gene), rows in merged.groupby(["chrom", "pos", "alt", "SYMBOL"]):
        d = score_rows(rows)
        d.update(chrom=chrom, pos=pos, ref=rows["ref"].iloc[0], alt=alt, gene=gene, gtype=rows["gtype"].iloc[0])
        out.append(d)
    vt = pd.DataFrame(out)
    if vt.empty:
        return vt
    vt["clinvar"] = None
    vt["clinvar_dis"] = None
    if use_clinvar:
        cand = vt[vt["score"] >= MIN_VARIANT_SCORE]
        targets = {(r.chrom, r.pos, r.ref, r.alt) for r in cand.itertuples()}
        try:
            info = clinvar_lookup(targets, clinvar_path)
        except Exception as e:  # noqa
            print("ClinVar недоступен, пропускаю:", e)
            info = {}
        for i, r in vt.iterrows():
            v = info.get((r.chrom, r.pos, r.ref, r.alt))
            if v and v[0]:
                vt.at[i, "clinvar"] = v[0]
                vt.at[i, "clinvar_dis"] = v[2]
                s = v[0].lower()
                if ("pathogenic" in s) and ("conflicting" not in s) and ("benign" not in s):
                    vt.at[i, "score"] = max(r.score, 0.95 * r.rarity)
                elif "benign" in s and "pathogenic" not in s:
                    vt.at[i, "score"] = r.score * 0.1
    return vt


# ----------------------------------------------------------------------------
# Сопоставление с фенотипом
# ----------------------------------------------------------------------------

def expand(terms, hpo):
    out = set()
    for t in terms:
        out |= hpo.ancestors(t)
    return out


def pheno_match(present, excluded, dterms, R):
    """Насколько болезнь (набор симптомов dterms) подходит пациенту. Возвращает (балл, детали)."""
    hpo, ic = R["hpo"], R["ic"]
    dexp = expand(dterms, hpo)
    pexp = expand(present, hpo)

    # со стороны пациента: насколько каждый его симптом "покрыт" болезнью
    num = den = 0.0
    credits = {}
    for p in present:
        icp = max(ic(p), 1e-6)
        common = hpo.ancestors(p) & dexp
        best = max((ic(a) for a in common), default=0.0)
        c = min(best / icp, 1.0)
        credits[p] = c
        num += icp * c
        den += icp
    patient_side = num / den if den else 0.0

    # со стороны болезни: сколько её типичных симптомов есть у пациента
    num = den = 0.0
    for d, w in dterms.items():
        icd = max(ic(d), 1e-6)
        best = max((ic(a) for a in hpo.ancestors(d) & pexp), default=0.0)
        num += w * icd * min(best / icd, 1.0)
        den += w * icd
    disease_side = num / den if den else 0.0

    # исключённые симптомы пациента, которые при этой болезни обычно бывают
    hits, hn, hd = [], 0.0, 0.0
    for e in excluded:
        ice = max(ic(e), 1e-6)
        wmax = 0.0
        for d, w in dterms.items():
            if e in hpo.ancestors(d):
                wmax = max(wmax, w)
        hd += ice
        hn += ice * wmax
        if wmax > 0:
            hits.append(e)
    penalty = 0.25 * (hn / hd) if hd else 0.0

    score = max(0.0, 0.75 * patient_side + 0.25 * disease_side - penalty)
    return score, {"credits": credits, "excluded_hits": hits,
                   "patient_side": patient_side, "disease_side": disease_side, "penalty": penalty}


def inheritance_factor(modes, gene_vars, sex, chrom_is_x):
    """Подходит ли набор вариантов к типу наследования. Возвращает (множитель, текст, V)."""
    q = gene_vars.sort_values("score", ascending=False)
    hom = q[q["gtype"].isin(["1/1", "1|1"])]
    het = q[~q["gtype"].isin(["1/1", "1|1"])]
    top = q["score"].tolist()
    best = top[0]
    results = []
    for m in (modes or ["?"]):
        if m == "AD":
            results.append((1.0, "доминантное наследование: достаточно одного изменённого аллеля - подходит", best))
        elif m == "AR":
            if len(hom):
                results.append((1.0, "рецессивное наследование: вариант в гомозиготе - подходит", hom["score"].max()))
            elif len(het) >= 2:
                v = (top[0] + top[1]) / 2
                results.append((0.9, "рецессивное наследование: два варианта в одном гене (возможная компаунд-гетерозигота; "
                                     "фазу по этим данным определить нельзя)", v))
            else:
                results.append((0.25, "рецессивное наследование, но найден только один гетерозиготный вариант - "
                                      "слабое соответствие", best))
        elif m == "XL":
            if chrom_is_x:
                results.append((1.0 if sex == "MALE" or len(hom) else 0.7, "Х-сцепленное наследование - учтено", best))
            else:
                results.append((0.1, "Х-сцепленное наследование, но ген не на X-хромосоме", best))
        else:
            results.append((0.6, "тип наследования для этой болезни неизвестен", best))
    return max(results, key=lambda x: x[0] * x[2])


# ----------------------------------------------------------------------------
# Главная функция анализа
# ----------------------------------------------------------------------------

def analyze_case(case_id, add_present=None, add_excluded=None, remove_terms=None,
                 sex=None, use_clinvar=True, top_n=10, verbose=True, save=True):
    """Анализирует один случай и возвращает таблицу кандидатов (рейтинг генов)."""
    R = load_resources()
    hpo = R["hpo"]
    ph = load_phenotype(case_id)
    present = [hpo.canon(t) for t in ph["present"]]
    excluded = [hpo.canon(t) for t in ph["excluded"]]
    labels = dict(ph["labels"])
    for t in (add_present or []):
        t = hpo.canon(t)
        if t not in present:
            present.append(t)
        excluded = [e for e in excluded if e != t]
        labels[t] = hpo.name(t)
    for t in (add_excluded or []):
        t = hpo.canon(t)
        if t not in excluded:
            excluded.append(t)
        present = [p for p in present if p != t]
        labels[t] = hpo.name(t)
    rm = {hpo.canon(t) for t in (remove_terms or [])}
    present = [p for p in present if p not in rm]
    excluded = [e for e in excluded if e not in rm]
    sex = (sex or ph["sex"]).upper()

    vt = build_variant_table(load_vep(case_id), load_genotypes(case_id), use_clinvar=use_clinvar)
    if vt.empty:
        print("Не нашлось ни одного размеченного варианта в генах.")
        return pd.DataFrame()
    n_total = len(vt)
    cand = vt[vt["score"] >= MIN_VARIANT_SCORE]

    rows = []
    for gene, gv in cand.groupby("gene"):
        is_x = (gv["chrom"].iloc[0] == "X")
        pairs = R["gene_pairs"].get(gene, [])
        best = None
        if not pairs:
            f, txt, v = inheritance_factor(["?"], gv, sex, is_x)
            best = dict(total=f * W_VARIANT * v, pheno=0.0, V=v, disease=None, inh_text="нет сведений о болезнях гена в HPO",
                        inh_factor=f, details=None)
        for dis in pairs:
            terms = R["pair_terms"][(gene, dis)]
            ps, det = pheno_match(present, excluded, terms, R)
            modes = sorted(R["pair_inh"].get((gene, dis), [])) or None
            f, txt, v = inheritance_factor(modes, gv, sex, is_x)
            total = f * (W_VARIANT * v + W_PHENO * ps)
            if best is None or total > best["total"]:
                best = dict(total=total, pheno=ps, V=v, disease=dis, inh_text=txt, inh_factor=f, details=det)
        top_v = gv.sort_values("score", ascending=False).iloc[0]
        best.update(gene=gene, n_variants=len(gv), variant=top_v, gv=gv)
        rows.append(best)

    rows.sort(key=lambda r: -r["total"])
    res = pd.DataFrame([{
        "ранг": i + 1, "ген": r["gene"], "итоговый_балл": round(r["total"], 3),
        "балл_варианта": round(r["V"], 3), "балл_фенотипа": round(r["pheno"], 3),
        "болезнь": (R["dnames"].get(r["disease"], r["disease"]) if r["disease"] else "нет данных"),
        "id_болезни": r["disease"], "вариант": f'chr{r["variant"].chrom}:{r["variant"].pos} {r["variant"].ref}>{r["variant"].alt}',
        "генотип": r["variant"].gtype, "последствие": r["variant"].consequence,
    } for i, r in enumerate(rows)])

    verdict, why = make_verdict(rows, len(present))
    report = build_report(case_id, rows, present, excluded, labels, sex, verdict, why, n_total, len(cand), top_n, R)
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        res.to_csv(os.path.join(OUT_DIR, f"{case_id}_ranking.csv"), index=False)
        open(os.path.join(OUT_DIR, f"{case_id}_report.txt"), "w", encoding="utf-8").write(report)
    if verbose:
        print(report)
    res.attrs["verdict"] = verdict
    return res


def make_verdict(rows, n_present=99):
    if not rows:
        return "нет кандидатов", "после фильтрации не осталось ни одного редкого вредоносного варианта"
    top = rows[0]["total"]
    second = rows[1]["total"] if len(rows) > 1 else 0
    margin = top - second
    if n_present < MIN_SYMPTOMS_FOR_CONFIDENCE and top >= POSSIBLE_SCORE:
        return "возможные кандидаты", (f"симптомов слишком мало ({n_present}), чтобы уверенно подтвердить гипотезу; "
                                       f"лучший балл {top:.2f}")
    if top >= CONFIDENT_SCORE and margin >= CONFIDENT_MARGIN:
        return "убедительный кандидат", f"лучший балл {top:.2f}, отрыв от второго места {margin:.2f}"
    if top >= POSSIBLE_SCORE:
        reason = "есть правдоподобный кандидат, но" + (
            f" отрыв от второго места мал ({margin:.2f}): несколько гипотез конкурируют" if margin < CONFIDENT_MARGIN
            else f" общий балл невысок ({top:.2f})")
        return "возможные кандидаты", reason
    return "убедительного кандидата нет", f"лучший балл всего {top:.2f}"


# ----------------------------------------------------------------------------
# Текстовое объяснение
# ----------------------------------------------------------------------------

def _pct(x):
    return f"{x * 100:.4f}%" if x < 0.01 else f"{x * 100:.2f}%"


def build_report(case_id, rows, present, excluded, labels, sex, verdict, why, n_total, n_cand, top_n, R):
    hpo = R["hpo"]
    L = []
    L.append("=" * 78)
    L.append(f"СЛУЧАЙ {case_id}   пол: {sex}   симптомов: {len(present)}, исключённых: {len(excluded)}")
    L.append(f"Вариантов в генах: {n_total}; после фильтров (редкие и вероятно вредные) осталось: {n_cand}")
    L.append(f"ВЫВОД СИСТЕМЫ: {verdict} ({why})")
    L.append("Метки: [БАЗА] = факт из базы данных, [ПРОГНОЗ] = компьютерное предсказание, "
             "[ГИПОТЕЗА] = вывод нашей системы")
    L.append("Только для исследований и обучения; не является клиническим диагнозом.")
    L.append("=" * 78)
    for i, r in enumerate(rows[:top_n], 1):
        v = r["variant"]
        dname = R["dnames"].get(r["disease"], r["disease"]) if r["disease"] else "нет данных"
        L.append(f"\n#{i}  {r['gene']}   итоговый балл {r['total']:.2f}   (вариант {r['V']:.2f}, фенотип {r['pheno']:.2f})")
        if i > 3:
            L.append(f"    болезнь: {dname}; вариант chr{v.chrom}:{v.pos} {v.ref}>{v.alt} ({v.gtype}), "
                     f"{CONS_RU.get(v.consequence, v.consequence)}")
            continue
        L.append(f"    болезнь, лучше всего подходящая к симптомам: {dname} ({r['disease']})")
        cons = CONS_RU.get(v.consequence, v.consequence)
        L.append(f"    Вариант: chr{v.chrom}:{v.pos} {v.ref}>{v.alt}, генотип {v.gtype}, {cons}"
                 + (f", {v.hgvsc}" if v.hgvsc else "") + (f", {v.hgvsp}" if v.hgvsp else ""))
        af = "не найден в gnomAD (крайне редкий)" if (v.af != v.af) else f"частота в gnomAD {_pct(v.af)}"
        L.append(f"    [БАЗА] {af}")
        if v.clinvar:
            L.append(f"    [БАЗА] ClinVar: {v.clinvar}" + (f" ({v.clinvar_dis})" if v.clinvar_dis else ""))
        pr = []
        if v.cadd == v.cadd:
            pr.append(f"CADD {v.cadd:.1f}")
        if v.revel == v.revel:
            pr.append(f"REVEL {v.revel:.2f}")
        if v.spliceai == v.spliceai:
            pr.append(f"SpliceAI {v.spliceai:.2f}")
        if pr:
            L.append("    [ПРОГНОЗ] " + ", ".join(pr))
        if r["details"]:
            cr = r["details"]["credits"]
            full = [labels.get(p, hpo.name(p)) for p, c in cr.items() if c >= 0.5]
            part = [labels.get(p, hpo.name(p)) for p, c in cr.items() if 0.15 <= c < 0.5]
            none = [labels.get(p, hpo.name(p)) for p, c in cr.items() if c < 0.15]
            L.append(f"    [БАЗА] симптомы пациента, согласующиеся с болезнью ({len(full)} из {len(cr)}): "
                     + (", ".join(full) if full else "нет"))
            if part:
                L.append("    частично согласуются: " + ", ".join(part))
            if none:
                L.append("    НЕ объясняются этой болезнью: " + ", ".join(none))
            if r["details"]["excluded_hits"]:
                L.append("    ВОЗМОЖНОЕ ПРОТИВОРЕЧИЕ: у пациента исключены симптомы, которые при этой болезни бывают (возможно, не у всех): "
                         + ", ".join(labels.get(e, hpo.name(e)) for e in r["details"]["excluded_hits"]))
        L.append(f"    [ГИПОТЕЗА] наследование: {r['inh_text']}")
        if r["n_variants"] > 1:
            L.append(f"    (в этом гене подозрительных вариантов: {r['n_variants']})")
    if verdict != "убедительный кандидат":
        L.append("\nЧТО МОГЛО БЫ УЛУЧШИТЬ ИНТЕРПРЕТАЦИЮ: более полный список симптомов, данные родителей "
                 "(фаза и происхождение вариантов), повторная классификация вариантов неопределённого значения, "
                 "более глубокий анализ нескодирующих областей.")
    return "\n".join(L)


def compare(case_id, **changes):
    """Дополнительная задача: показывает, как меняется рейтинг после изменений в фенотипе."""
    before = analyze_case(case_id, verbose=False, save=False)
    after = analyze_case(case_id, verbose=False, save=False, **changes)
    b = before.set_index("ген")["ранг"]
    out = after.head(10)[["ранг", "ген", "итоговый_балл", "болезнь"]].copy()
    out["ранг_раньше"] = out["ген"].map(b)
    out["сдвиг"] = out["ранг_раньше"] - out["ранг"]
    print("Вывод до изменений :", before.attrs.get("verdict"))
    print("Вывод после        :", after.attrs.get("verdict"))
    return out


def run_all(case_ids=None, use_clinvar=True):
    """Прогоняет все найденные случаи и выдаёт сводную таблицу."""
    if case_ids is None:
        files = glob.glob(os.path.join(DATA_DIR, "**", "case_*_phenotype.json"), recursive=True)
        case_ids = sorted({re.search(r"(case_\d+)", f).group(1) for f in files})
    rows = []
    for c in case_ids:
        try:
            r = analyze_case(c, verbose=False, use_clinvar=use_clinvar)
            if r.empty:
                rows.append({"случай": c, "вывод": "нет кандидатов"})
                continue
            t = r.iloc[0]
            rows.append({"случай": c, "вывод": r.attrs["verdict"], "топ-1 ген": t["ген"], "балл": t["итоговый_балл"],
                         "болезнь": t["болезнь"], "топ-2": r.iloc[1]["ген"] if len(r) > 1 else "",
                         "топ-3": r.iloc[2]["ген"] if len(r) > 2 else ""})
        except Exception as e:  # noqa
            rows.append({"случай": c, "вывод": f"ошибка: {e}"})
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    return df
