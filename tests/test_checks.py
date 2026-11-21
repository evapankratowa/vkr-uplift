"""Команда `check` (тикет T7): прогон карточек, отчёт в checks/, код возврата.

Синтетика прогоняется на малой конфигурации (2000 клиентов, 100 прогонов A/A),
реальные выгрузки не нужны: проверки сырых данных и сетевых наборов обязаны
пропускаться жёлтым «пропущено: нет входа», а не падать. Красный сценарий
проверяется подложкой битого файла X5, зелёный путь карточек выгрузки —
фикстурным CSV со статистиками DATA_PASSPORT (50 000 строк, 42 колонки).

Прогонщик (CheckRunner) покрыт отдельно: вердикты пяти исходов, код возврата,
машиночитаемый json и markdown-таблица.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from uplift_system import cli
from uplift_system.checks import cards, runner
from uplift_system.checks.runner import Check, CheckRunner, Outcome

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _toy_check(check_id: str, fn, on_fail: str = "СТОП") -> Check:
    """Карточка-заглушка для юнит-тестов прогонщика."""
    return Check(
        id=check_id,
        what="тестовая проверка",
        compare="значение против эталона",
        expect="совпадает",
        on_fail=on_fail,
        fn=fn,
    )


def test_runner_assigns_verdicts_and_exit_code():
    """Пять исходов прогона: pass, fail-стоп, fail-пометка, skip, исключение."""
    probe = CheckRunner()
    probe.add(_toy_check("ok", lambda: Outcome.passed(1.0, "в пределах допуска")))
    probe.add(_toy_check("stop", lambda: Outcome.failed(2.0, "критично")))
    probe.add(
        _toy_check(
            "mark",
            lambda: Outcome.failed(3.0, "не критично"),
            on_fail="ИДУ ДАЛЬШЕ С ПОМЕТКОЙ",
        )
    )
    probe.add(_toy_check("skip", lambda: Outcome.skipped("пропущено: нет входа")))
    probe.add(_toy_check("boom", lambda: 1 / 0))

    probe.run()

    by_id = {check.id: check for check in probe.checks}
    assert by_id["ok"].verdict == "сошлось"
    assert by_id["ok"].result == 1.0
    assert by_id["stop"].verdict == "стоп"
    assert by_id["mark"].verdict == "разошлось, объяснимо"
    assert by_id["skip"].verdict == "не выполнена"
    assert by_id["boom"].verdict == "стоп"
    assert "ZeroDivisionError" in by_id["boom"].note
    assert probe.counts() == {"green": 1, "yellow": 2, "red": 2}
    assert probe.exit_code == 1


def test_runner_without_red_exits_zero():
    """Жёлтые пропуски не роняют прогон: код возврата 0."""
    probe = CheckRunner()
    probe.add(_toy_check("ok", lambda: Outcome.passed("x")))
    probe.add(_toy_check("skip", lambda: Outcome.skipped("пропущено: нет входа")))
    probe.run()
    assert probe.exit_code == 0


def test_runner_writes_json_and_markdown(tmp_path, monkeypatch):
    """to_json пишет machine-readable отчёт с вердиктами; to_markdown — таблицу."""
    monkeypatch.setattr(runner, "CHECKS", tmp_path)  # дефолтный путь — во временную папку
    probe = CheckRunner()
    probe.add(_toy_check("ok", lambda: Outcome.passed(0.03, "в пределах допуска")))
    probe.add(_toy_check("skip", lambda: Outcome.skipped("пропущено: нет входа")))
    probe.run()

    path = probe.to_json()
    assert path == tmp_path / "checks.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["green"] == 1
    assert payload["summary"]["yellow"] == 1
    assert payload["summary"]["red"] == 0
    assert payload["summary"]["exit_code"] == 0
    first = payload["checks"][0]
    for key in ("id", "what", "compare", "expect", "result", "verdict", "color", "note"):
        assert key in first
    assert first["verdict"] == "сошлось"
    assert first["color"] == "green"

    table = probe.to_markdown()
    assert "| Проверка | Что с чем | Ожидали | Получили | Вердикт |" in table
    assert "Итог" in table
    assert "Код возврата: 0" in table
    assert "сошлось (зелёная)" in table
    assert "не выполнена (жёлтая)" in table


@pytest.fixture()
def check_configs(tmp_path: Path) -> tuple[Path, Path]:
    """Маленькие конфиги команды check: синтетика 2000 клиентов, A/A на 100 прогонов.

    Источник ковариат направлен в несуществующий файл: выгрузки кэшбэка на
    машине разработчика может и не быть, тест обязан быть герметичным.
    """
    sim = yaml.safe_load((PROJECT_DIR / "configs" / "sim_base.yaml").read_text(encoding="utf-8"))
    sim["sample"]["n_clients"] = 2000
    sim["covariates"]["source"] = str(tmp_path / "cashback_raw_2022-11-15.csv")
    exp = yaml.safe_load((PROJECT_DIR / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
    exp["validation"] = {"aa_test": True, "aa_runs": 100, "holdout_share": 0.3}
    sim_path = tmp_path / "sim_small.yaml"
    exp_path = tmp_path / "exp_small.yaml"
    sim_path.write_text(yaml.safe_dump(sim, allow_unicode=True), encoding="utf-8")
    exp_path.write_text(yaml.safe_dump(exp, allow_unicode=True), encoding="utf-8")
    return sim_path, exp_path


@pytest.fixture()
def empty_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Пустые data/raw, models_store и output/tables: входные проверки жёлтые.

    Подмена обязательна даже в репозитории без данных: на машине разработчика
    артефакты волн 1–3 могут лежать в output/tables, и карточки их подхватят.
    """
    monkeypatch.setattr(cards, "DATA_RAW", tmp_path / "data_raw")
    monkeypatch.setattr(cards, "MODELS_STORE", tmp_path / "models_store")
    monkeypatch.setattr(cards, "OUT_TABLES", tmp_path / "out_tables")


def _run_check(sim_path: Path, exp_path: Path, out_dir: Path) -> int:
    """Запуск команды check на переданных конфигах."""
    return cli.main(
        [
            "check",
            "--config",
            str(sim_path),
            "--experiment",
            str(exp_path),
            "--out-dir",
            str(out_dir),
        ]
    )


def test_check_help_informative(capsys):
    """--help команды check описывает конфиги и выходной каталог."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["check", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--config", "--experiment", "--out-dir"):
        assert flag in out


def test_check_command_writes_reports_and_exits_zero(
    tmp_path, monkeypatch, check_configs, empty_inputs, capsys
):
    """Прогон без входов: синтетика зелёная, остальное жёлтое «нет входа», код 0."""
    sim_path, exp_path = check_configs
    out_dir = tmp_path / "checks"
    code = _run_check(sim_path, exp_path, out_dir)
    assert code == 0

    payload = json.loads((out_dir / "checks.json").read_text(encoding="utf-8"))
    assert payload["summary"]["red"] == 0
    assert payload["summary"]["exit_code"] == 0
    by_verdict: dict[str, list[str]] = {}
    for check in payload["checks"]:
        by_verdict.setdefault(check["verdict"], []).append(check["id"])
    assert {"synthetic.ate", "synthetic.aa"} <= set(by_verdict["сошлось"])
    for check_id in (
        "cashback.rows",
        "cashback.rates",
        "x5.control",
        "hillstrom.control",
        "models.manifest",
        "models.files",
        "economics.sms",
        "evaluate.columns",
        "segments.exports",
    ):
        assert check_id in by_verdict["не выполнена"]
    skipped = [c for c in payload["checks"] if c["verdict"] == "не выполнена"]
    assert all("пропущено: нет входа" in c["note"] for c in skipped)

    reports = list(out_dir.glob("checks_*.md"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "Код возврата: 0" in text
    assert "пропущено: нет входа" in text
    assert "synthetic.ate" in text

    out = capsys.readouterr().out
    assert "Отчёт проверок" in out
    assert "Итог" in out


def test_check_command_exits_one_on_red_control_number(
    tmp_path, monkeypatch, check_configs, empty_inputs
):
    """Битый файл X5 (30 строк вместо 200 039) — красная проверка и код возврата 1."""
    data_raw = tmp_path / "data_raw"
    data_raw.mkdir()
    fake = pd.DataFrame({"treatment_flg": [0, 1] * 15, "target": [1, 0] * 15})
    fake.to_csv(data_raw / "x5_uplift_train_raw.csv.gz", index=False, compression="gzip")
    monkeypatch.setattr(cards, "DATA_RAW", data_raw)  # поверх empty_inputs

    sim_path, exp_path = check_configs
    code = _run_check(sim_path, exp_path, tmp_path / "checks")
    assert code == 1

    payload = json.loads((tmp_path / "checks" / "checks.json").read_text(encoding="utf-8"))
    assert payload["summary"]["red"] == 1
    assert payload["summary"]["exit_code"] == 1
    x5 = next(check for check in payload["checks"] if check["id"] == "x5.control")
    assert x5["verdict"] == "стоп"
    assert x5["color"] == "red"
    assert x5["result"]["строк"] == 30


def _write_fake_cashback(path: Path) -> None:
    """Выгрузка кэшбэка со статистиками DATA_PASSPORT: 50 000 строк, 42 колонки.

    Доли ненулевых и медианы/средние задаются детерминированным паттерном по
    индексу строки; распределения не нужны — карточки смотрят агрегаты, сырые
    строки тест только пишет в файл, никуда больше не передавая.
    """
    n = 50_000
    index = pd.Index(range(1, n + 1), name="client_id")
    row = pd.Series(range(n))
    mod = (row % 10_000).to_numpy()
    low = (row % 10).to_numpy()

    data: dict[str, pd.Series] = {}
    for name in cards.SPEND_CATEGORIES:
        data[name] = pd.Series(0, index=index)  # окно «после» категорий карточки не читают
        data[f"{name}_before"] = pd.Series(0, index=index)

    merch = pd.Series(0.0, index=index)
    merch_pos = mod < 4_795  # доля 0.4795
    merch.loc[merch_pos & (low < 9)] = 206.0
    merch.loc[merch_pos & (low >= 9)] = 4_466.5
    data["cb_merch_last_3_month"] = merch

    bank = pd.Series(0.0, index=index)
    bank_pos = mod < 3_838  # доля 0.3838
    bank.loc[bank_pos & (low < 9)] = 120.0
    bank.loc[bank_pos & (low >= 9)] = 2_529.7
    data["cb_bank_last_3_month"] = bank

    data["cb_merch_before"] = pd.Series((mod < 6_737).astype(float), index=index)  # 0.6737
    data["cb_bank_before"] = pd.Series((mod < 6_569).astype(float), index=index)  # 0.6569

    communication = pd.Series(0, index=index)
    communication.loc[mod < 6_572] = 1  # доля получавших 0.6572, медиана 1
    communication.loc[(row % 9_973 == 0).to_numpy()] = 18  # максимум 18
    data["cnt_communication"] = communication
    data["cnt_view"] = pd.Series(0, index=index)
    data["cnt_activation"] = pd.Series(0, index=index)

    pd.DataFrame(data).to_csv(path, index=True)


def test_check_command_validates_cashback_dump(tmp_path, monkeypatch, check_configs, empty_inputs):
    """Фикстурная выгрузка с контрольными числами паспорта: шесть зелёных карточек."""
    sim_path, exp_path = check_configs
    sim = yaml.safe_load(sim_path.read_text(encoding="utf-8"))
    dump_path = tmp_path / "cashback_raw_2022-11-15.csv"
    _write_fake_cashback(dump_path)
    sim["covariates"]["source"] = str(dump_path)
    sim_path.write_text(yaml.safe_dump(sim, allow_unicode=True), encoding="utf-8")

    code = _run_check(sim_path, exp_path, tmp_path / "checks")
    assert code == 0

    payload = json.loads((tmp_path / "checks" / "checks.json").read_text(encoding="utf-8"))
    cashback = {
        check["id"]: check for check in payload["checks"] if check["id"].startswith("cashback.")
    }
    assert len(cashback) == 6
    assert all(check["verdict"] == "сошлось" for check in cashback.values())
    assert cashback["cashback.rows"]["result"] == {"строк": 50_000, "client_id уникален": True}


def test_check_command_validates_pipeline_artifacts(
    tmp_path, monkeypatch, check_configs, empty_inputs
):
    """Артефакты волн 1–3 на месте: экономика, колонки evaluate, модели, сегменты — зелёные."""
    out_tables = tmp_path / "out_tables"
    out_tables.mkdir()
    prefix = cards.PROJECT_PREFIX
    econ = pd.DataFrame(
        {
            "канал": ["push", "sms", "sms", "sms", "call"],
            "цена_контакта_руб": [0.125, 3.2, 5.75, 30.0, 45.0],
        }
    )
    econ.to_csv(out_tables / f"{prefix}_economics_2026-11-21.csv", index=False)
    arch = pd.DataFrame(columns=cards.ARCHITECTURE_COLUMNS)
    arch.to_csv(out_tables / f"{prefix}_architectures_2026-11-21.csv", index=False)
    pd.DataFrame({"client_id": [1], "uplift_score": [0.01]}).to_csv(
        out_tables / f"{prefix}_segment_demo.csv", index=False
    )
    pd.DataFrame({"client_id": [2], "uplift_score": [None]}).to_csv(
        out_tables / f"{prefix}_segment_ab_demo.csv", index=False
    )
    pd.DataFrame({"client_id": [1], "uplift_score": [0.01]}).to_parquet(
        out_tables / f"{prefix}_segment_demo.parquet", index=False
    )

    models_store = tmp_path / "models_store"
    models_store.mkdir()
    models_yaml = PROJECT_DIR / "configs" / "models.yaml"
    config_hash = hashlib.sha256(models_yaml.read_bytes()).hexdigest()
    manifest = {"date": "2026-11-21", "config": str(models_yaml), "config_hash": config_hash}
    (models_store / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    cfg_models = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    enabled = [
        item["name"]
        for item in cfg_models.get("architectures", []) + cfg_models.get("baselines", [])
        if item.get("enabled", True)
    ]
    for name in enabled:
        (models_store / f"{name}.joblib").touch()

    sim_path, exp_path = check_configs
    code = _run_check(sim_path, exp_path, tmp_path / "checks")
    assert code == 0

    payload = json.loads((tmp_path / "checks" / "checks.json").read_text(encoding="utf-8"))
    by_id = {check["id"]: check for check in payload["checks"]}
    for check_id in (
        "models.manifest",
        "models.files",
        "economics.sms",
        "evaluate.columns",
        "segments.exports",
    ):
        assert by_id[check_id]["verdict"] == "сошлось", by_id[check_id]["note"]
    assert by_id["models.manifest"]["result"]["хеш манифеста"] == config_hash[:12]


def test_check_command_red_when_manifest_stale(tmp_path, monkeypatch, check_configs, empty_inputs):
    """Манифест с чужим хешем конфига — красная карточка models.manifest и код 1."""
    models_store = tmp_path / "models_store"
    models_store.mkdir()
    manifest = {"date": "2026-01-01", "config_hash": "0" * 64}
    (models_store / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(cards, "MODELS_STORE", models_store)  # поверх empty_inputs

    sim_path, exp_path = check_configs
    code = _run_check(sim_path, exp_path, tmp_path / "checks")
    assert code == 1

    payload = json.loads((tmp_path / "checks" / "checks.json").read_text(encoding="utf-8"))
    manifest_check = next(c for c in payload["checks"] if c["id"] == "models.manifest")
    assert manifest_check["verdict"] == "стоп"
