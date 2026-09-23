"""Browser coverage for the canonical CSV and Excel import lifecycle."""

from __future__ import annotations

from pathlib import Path
import uuid

import pytest
from playwright.sync_api import Page, expect

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CSV_FIXTURE = REPOSITORY_ROOT / "assets" / "QA" / "adsmod-dataset-csv-20260916.csv"
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def _browser_errors(page: Page) -> tuple[list[str], list[str], list[str]]:
    console_errors: list[str] = []
    page_errors: list[str] = []
    api_errors: list[str] = []
    page.on(
        "console",
        lambda message: console_errors.append(message.text)
        if message.type == "error"
        else None,
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "response",
        lambda response: api_errors.append(
            f"{response.status} {response.url}"
        )
        if "/api/v1/datasets" in response.url and response.status >= 400
        else None,
    )
    return console_errors, page_errors, api_errors


def _import_inspect_reload_and_delete(
    page: Page,
    fixture_path: Path,
    dataset_name: str,
) -> None:
    expect(page.locator("input[type='file']")).to_have_attribute(
        "accept", ".csv,.xls,.xlsx"
    )
    page.locator("input[type='file']").set_input_files(str(fixture_path))
    wizard = page.get_by_role("dialog")
    expect(wizard).to_be_visible()
    expect(wizard).to_contain_text(fixture_path.name)
    expect(wizard.locator(".import-preview-table thead th")).to_have_count(9)

    page.get_by_role("button", name="Review mapping").click()
    page.get_by_label("Dataset name").fill(dataset_name)
    page.get_by_role("button", name="Validate and preview series").click()
    summary = wizard.locator(".import-summary-grid strong")
    expect(summary.nth(0)).to_have_text("valid")
    expect(summary.nth(1)).to_have_text("2")
    expect(summary.nth(2)).to_have_text("6")

    page.get_by_role("button", name="Save validated dataset").click()
    expect(
        wizard.get_by_role("heading", name=f"{dataset_name} is ready")
    ).to_be_visible()
    page.get_by_role("button", name="Done").click()
    expect(wizard).to_be_hidden()

    page.reload()
    dataset_card = page.locator(".dataset-record").filter(has_text=dataset_name)
    expect(dataset_card).to_be_visible()
    expect(dataset_card).to_contain_text("2 experiments")
    expect(dataset_card).to_contain_text("6 observations")
    dataset_card.get_by_role("button", name="Select").click()
    expect(page.get_by_text("Detected columns", exact=True)).to_be_visible()
    expect(page.locator(".dataset-experiment")).to_have_count(2)

    page.locator(".dataset-experiment").nth(1).click()
    expect(page.get_by_text("3 rows", exact=True)).to_be_visible()
    expect(
        page.get_by_role("table", name="Persisted observations").locator(
            "tbody tr"
        )
    ).to_have_count(3)

    page.reload()
    dataset_card = page.locator(".dataset-record").filter(has_text=dataset_name)
    expect(dataset_card).to_be_visible()
    dataset_card.get_by_role("button", name="Select").click()
    expect(page.get_by_text("Detected columns", exact=True)).to_be_visible()
    expect(page.locator(".dataset-experiment")).to_have_count(2)

    with page.expect_response(
        lambda response: response.request.method == "DELETE"
        and "/api/v1/datasets/" in response.url
    ) as delete_response:
        dataset_card.get_by_role("button", name="Delete").click()
    assert delete_response.value.status == 204
    expect(dataset_card).to_be_hidden()

    page.reload()
    expect(
        page.locator(".dataset-record").filter(has_text=dataset_name)
    ).to_have_count(0)


def test_csv_import_persists_inspection_experiment_switch_and_deletion(
    page_context: Page,
) -> None:
    page = page_context
    page.goto("http://127.0.0.1:5173/datasets")
    console_errors, page_errors, api_errors = _browser_errors(page)
    expect(page.get_by_text("Online", exact=True)).to_be_visible()

    _import_inspect_reload_and_delete(
        page,
        CSV_FIXTURE,
        f"ADS-T2-01 CSV {uuid.uuid4().hex[:8]}",
    )
    assert not console_errors, "\n".join(console_errors)
    assert not page_errors, "\n".join(page_errors)
    assert not api_errors, "\n".join(api_errors)


@pytest.mark.parametrize("extension", [".xls", ".xlsx"])
def test_excel_import_persists_inspection_and_reload(
    page_context: Page,
    extension: str,
) -> None:
    page = page_context
    page.goto("http://127.0.0.1:5173/datasets")
    console_errors, page_errors, api_errors = _browser_errors(page)
    expect(page.get_by_text("Online", exact=True)).to_be_visible()

    _import_inspect_reload_and_delete(
        page,
        FIXTURES_DIR / f"sample_adsorption{extension}",
        f"ADS-T2-02 Excel {extension[1:]} {uuid.uuid4().hex[:8]}",
    )
    assert not console_errors, "\n".join(console_errors)
    assert not page_errors, "\n".join(page_errors)
    assert not api_errors, "\n".join(api_errors)
