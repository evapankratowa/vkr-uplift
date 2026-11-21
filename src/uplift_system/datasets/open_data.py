"""Открытые наборы с настоящей рандомизацией.

Роль в работе: проверить, что выводы, полученные на симуляции, держатся на
настоящем эксперименте. Истинного CATE здесь нет и быть не может — он
ненаблюдаем, — поэтому доступны только ранжирующие метрики.

  Hillstrom  64 000 клиентов, три ветки (Mens / Womens / без письма), RCT.
             433 КБ сжатым. Быстрый набор для смоук-теста конвейера.
  X5 RetailHero
             200 039 клиентов в обучающей части, SMS-кампания, RCT,
             доля теста 0.5, доля отклика 0.62.

**Про X5 качаются только два файла из трёх.** `purchases.csv` — это 45,8 млн
строк истории покупок, 4,17 ГБ в распакованном виде. Для признаков клиента её
можно агрегировать, но это отдельная работа и отдельные часы; базовый набор
признаков берётся из `clients.csv`. Если история покупок понадобится, скачивать
её нужно осознанно, а не заодно.

Скачивание идёт по прямым ссылкам scikit-uplift (там же лежат наборы, которые
отдаёт `sklift.datasets.fetch_*`). Сам `sklift` остаётся в проекте как эталон
метрик — см. решение №8.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .base import UpliftDataset

SOURCES = {
    "hillstrom": {
        "url": "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz",
        "filename": "hillstrom_raw.csv.gz",
        "approx_mb": 0.5,
    },
    "x5_clients": {
        "url": "https://sklift.s3.eu-west-2.amazonaws.com/clients.csv.gz",
        "filename": "x5_clients_raw.csv.gz",
        "approx_mb": 12,
    },
    "x5_train": {
        "url": "https://sklift.s3.eu-west-2.amazonaws.com/uplift_train.csv.gz",
        "filename": "x5_uplift_train_raw.csv.gz",
        "approx_mb": 2,
    },
    # не качается по умолчанию: 647 МБ сжатым, 4.17 ГБ распакованным
    "x5_purchases": {
        "url": "https://sklift.s3.eu-west-2.amazonaws.com/purchases.csv.gz",
        "filename": "x5_purchases_raw.csv.gz",
        "approx_mb": 647,
    },
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

# Признаки X5, на которые повлияла сама кампания: обусловливаться на них нельзя.
POST_TREATMENT_PREFIXES = ("days_since_first_redeem_date", "first_redeem_date_missing")


def download(name: str, data_dir: Path | None = None, force: bool = False) -> Path:
    """Скачивает набор в data/raw, если его там ещё нет. Возвращает путь к файлу.

    Сеть нужна прямая: доступ к S3 может быть закрыт корпоративной политикой
    (прокси отвечает 403 на CONNECT). В этом случае файл кладётся руками — имя
    и ссылка печатаются в сообщении об ошибке.
    """
    assert name in SOURCES, f"неизвестный набор {name}, есть {sorted(SOURCES)}"
    source = SOURCES[name]
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / source["filename"]

    if dest.exists() and not force:
        print(f"{name}: уже скачан, {dest.stat().st_size / 1e6:.1f} МБ -> {dest.name}")
        return dest

    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError("нужен requests: pip install requests") from exc

    print(f"{name}: качаю ~{source['approx_mb']} МБ с {source['url']}")
    try:
        response = requests.get(source["url"], stream=True, timeout=60)
        response.raise_for_status()
        with dest.open("wb") as fh:
            shutil.copyfileobj(response.raw, fh)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"не удалось скачать {name}. Если это 403 от прокси — доступ к хосту "
            f"закрыт политикой сети, обходить её не нужно. Положи файл руками:\n"
            f"  {source['url']}\n  -> {dest}"
        ) from exc
    print(f"{name}: готово, {dest.stat().st_size / 1e6:.1f} МБ -> {dest.name}")
    return dest


def _read_csv_any(path: Path, **kwargs) -> pd.DataFrame:
    """Читает csv или csv.gz, не полагаясь на расширение."""
    with path.open("rb") as fh:
        magic = fh.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rt", encoding="utf8") as fh:  # type: ignore[operator]
        return pd.read_csv(fh, **kwargs)


def _one_hot(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Числовая матрица признаков: все нечисловые колонки в индикаторы.

    Не используем встроенную работу CatBoost с категориями, чтобы все
    архитектуры получали ровно одну и ту же матрицу признаков.

    Нечисловые колонки определяются по типу, а не по переданному списку:
    в pandas 3 строки хранятся в StringDtype, а не в object, и проверка
    `dtype == object` их не ловит — колонка молча уезжает в astype(float)
    и падает на первом же значении.
    """
    hint = [c for c in (columns or []) if c in frame.columns]
    detected = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    present = list(dict.fromkeys(hint + detected))
    numeric = frame.drop(columns=present)
    if present:
        dummies = pd.get_dummies(frame[present], prefix=present, dtype=float)
        numeric = pd.concat([numeric, dummies], axis=1)
    return numeric.astype(float)


def load_hillstrom(
    data_dir: Path | None = None,
    treatment_arm: str = "Womens E-Mail",
    target_col: str = "visit",
    download_if_missing: bool = True,
) -> UpliftDataset:
    """Hillstrom MineThatData: одна ветка рассылки против «без письма».

    Из трёх групп берутся две: выбранная ветка как тест и «No E-Mail» как
    контроль. Третья ветка выбрасывается целиком — рандомизация от этого не
    страдает, потому что распределение по трём группам было случайным, и любая
    пара групп остаётся рандомизированным сравнением.

    target_col: visit (доля около 0.15) или conversion (около 0.009).
    """
    path = Path(data_dir or DEFAULT_DATA_DIR) / SOURCES["hillstrom"]["filename"]
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"нет файла {path}")
        path = download("hillstrom", data_dir)

    raw = _read_csv_any(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    assert "segment" in raw.columns, f"нет колонки segment, есть {list(raw.columns)}"
    assert target_col in raw.columns, f"нет колонки {target_col}"

    arm = treatment_arm.strip().lower()
    segments = raw["segment"].astype(str).str.strip().str.lower()
    control_mask = segments == "no e-mail"
    treat_mask = segments == arm
    assert treat_mask.any(), f"ветка {treatment_arm} не найдена, есть {sorted(segments.unique())}"

    kept = raw[control_mask | treat_mask].reset_index(drop=True)
    treatment = pd.Series(
        np.where(segments[control_mask | treat_mask].to_numpy() == arm, 1, 0), name="treatment"
    )
    outcome = kept[target_col].astype(int)

    feature_cols = [
        c
        for c in (
            "recency",
            "history_segment",
            "history",
            "mens",
            "womens",
            "zip_code",
            "newbie",
            "channel",
        )
        if c in kept.columns
    ]
    features = _one_hot(kept[feature_cols], ["history_segment", "zip_code", "channel"])
    index = pd.Index(np.arange(len(kept)), name="client_id")
    features.index = index

    return UpliftDataset(
        features=features,
        id=pd.Series(index.to_numpy(), index=index),
        treatment=pd.Series(treatment.to_numpy(), index=index),
        outcomes={"purchase": pd.Series(outcome.to_numpy(), index=index)},
        name=f"hillstrom_{arm.replace(' ', '_')}_{target_col}",
        meta={
            "source": "hillstrom",
            "treatment_arm": treatment_arm,
            "target_col": target_col,
            "treat_prob": float(treatment.mean()),
            "base_rate_control": float(outcome[treatment.to_numpy() == 0].mean()),
            "randomized": True,
            "has_ground_truth": False,
        },
    )


def load_x5(
    data_dir: Path | None = None,
    download_if_missing: bool = True,
    include_post_treatment: bool = False,
) -> UpliftDataset:
    """X5 RetailHero: SMS-кампания, рандомизированный эксперимент.

    Признаки — из `clients.csv`: возраст, пол, давность регистрации. История
    покупок (`purchases.csv`, 4,17 ГБ) не используется: агрегировать её —
    отдельная работа, см. докстринг модуля.

    **Дата первого списания бонусов выброшена как пост-воздействие.**
    `first_redeem_date` — дата, на которую повлияла сама кампания: SMS про
    кэшбэк подталкивает клиента списать бонусы. Признак и его флаг пропуска
    оказываются функцией воздействия, и обусловливаться на них нельзя — это
    блокирует часть пути от коммуникации к покупке. В данных это видно прямо:
    доля назначенных в тест по децилям давности списания даёт |z| до 14,9 при
    границе по Бонферрони 3,2, причём отклонения сидят только в двух крайних
    децилях и в противоположные стороны — след сдвига сроков списания кампанией.
    Давность регистрации, возраст и пол сбалансированы (|z| до 2,6).

    include_post_treatment=True возвращает наивный вариант со всеми признаками —
    нужен, только чтобы показать разницу, в расчёты он не идёт.
    """
    data_dir = Path(data_dir or DEFAULT_DATA_DIR)
    paths = {}
    for name in ("x5_clients", "x5_train"):
        path = data_dir / SOURCES[name]["filename"]
        if not path.exists():
            if not download_if_missing:
                raise FileNotFoundError(f"нет файла {path}")
            path = download(name, data_dir)
        paths[name] = path

    clients = _read_csv_any(paths["x5_clients"])
    train = _read_csv_any(paths["x5_train"])
    clients.columns = [c.strip().lower() for c in clients.columns]
    train.columns = [c.strip().lower() for c in train.columns]

    id_col = next((c for c in ("client_id", "customer_id") if c in train.columns), None)
    assert id_col is not None, f"не нашла ключ клиента в {list(train.columns)}"
    assert "treatment_flg" in train.columns, "нет treatment_flg"
    assert "target" in train.columns, "нет target"

    merged = train.merge(clients, on=id_col, how="inner", validate="one_to_one")
    lost = len(train) - len(merged)
    if lost:
        print(f"внимание: {lost} клиентов из обучающей части не нашли пары в clients.csv")

    # даты в числа: абсолютная дата модели не нужна, нужна давность.
    # Опорная дата ОДНА на все колонки и считается до цикла: если брать максимум
    # нарастающим, первая колонка отсчитывается от своего максимума, а следующая —
    # от общего, и две «давности» оказываются на разных базах.
    date_cols = [c for c in merged.columns if "date" in c]
    parsed_dates = {c: pd.to_datetime(merged[c], errors="coerce") for c in date_cols}
    reference = max((s.max() for s in parsed_dates.values() if s.notna().any()), default=None)
    for column, parsed in parsed_dates.items():
        merged[f"days_since_{column}"] = (
            (reference - parsed).dt.days if reference is not None else np.nan
        )
        merged[f"{column}_missing"] = parsed.isna().astype(float)
    merged = merged.drop(columns=date_cols)
    if reference is not None:
        print(f"опорная дата для давности: {reference.date()} (максимум по всем датам)")

    # Признаки, затронутые воздействием, по умолчанию выбрасываются.
    dropped = []
    if not include_post_treatment:
        for column in list(merged.columns):
            if column.startswith(POST_TREATMENT_PREFIXES):
                merged = merged.drop(columns=column)
                dropped.append(column)
        if dropped:
            print("выброшены признаки, затронутые воздействием: " + ", ".join(dropped))

    outcome = merged["target"].astype(int)
    treatment = merged["treatment_flg"].astype(int)
    features_raw = merged.drop(columns=[id_col, "target", "treatment_flg"])
    features = _one_hot(features_raw).fillna(-1.0)

    index = pd.Index(merged[id_col].to_numpy(), name=id_col)
    features.index = index

    return UpliftDataset(
        features=features,
        id=pd.Series(index.to_numpy(), index=index),
        treatment=pd.Series(treatment.to_numpy(), index=index),
        outcomes={"purchase": pd.Series(outcome.to_numpy(), index=index)},
        name="x5_retailhero",
        meta={
            "source": "x5_retailhero",
            "treat_prob": float(treatment.mean()),
            "base_rate_control": float(outcome[treatment.to_numpy() == 0].mean()),
            "randomized": True,
            "has_ground_truth": False,
            "purchases_used": False,
            "post_treatment_features_dropped": dropped,
            "reference_date": str(reference.date()) if reference is not None else None,
        },
    )


LOADERS = {"hillstrom": load_hillstrom, "x5": load_x5}


def main(argv: list[str] | None = None) -> int:
    """CLI: скачать наборы в data/raw и напечатать их профиль."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hillstrom", "x5", "all"], default="all")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--with-purchases",
        action="store_true",
        help="скачать также purchases.csv X5 — 647 МБ сжатым, 4.17 ГБ распакованным",
    )
    args = parser.parse_args(argv)

    names = ["hillstrom", "x5"] if args.dataset == "all" else [args.dataset]
    for name in names:
        dataset = LOADERS[name](data_dir=args.data_dir)
        meta = dataset.meta or {}
        print(
            f"\n{dataset.name}: клиентов {dataset.n}, признаков {dataset.features.shape[1]},"
            f" доля теста {meta.get('treat_prob'):.4f},"
            f" базовый отклик в контроле {meta.get('base_rate_control'):.4f}"
        )
    if args.with_purchases:
        download("x5_purchases", args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
