# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

"""Browser coverage for the canonical CSV and Excel import lifecycle."""

from __future__ import annotations

from pathlib import Path
import re
import uuid

import pytest
from playwright.sync_api import Page, expect

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CSV_FIXTURE = REPOSITORY_ROOT / "assets" / "QA" / "adsmod-dataset-csv-20260916.csv"
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


###############################################################################
def _browser_errors(page: Page) -> tuple[list[str], list[str], list[str]]:
    console_errors: list[str] = []
    page_errors: list[str] = []
    api_errors: list[str] = []
    page.on(
        "console",
        lambda message: (
            console_errors.append(message.text) if message.type == "error" else None
        ),
    )
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.on(
        "response",
        lambda response: (
            api_errors.append(f"{response.status} {response.url}")
            if "/api/v1/datasets" in response.url and response.status >= 400
            else None
        ),
    )
    return console_errors, page_errors, api_errors


###############################################################################
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
    summary = wizard.locator(".validation-summary strong")
    expect(summary.nth(0)).to_have_text("Valid")
    expect(summary.nth(1)).to_have_text("2")
    expect(summary.nth(2)).to_have_text("6")

    page.get_by_role("button", name="Save validated dataset").click()
    expect(
        wizard.get_by_role("heading", name=f"{dataset_name} is ready")
    ).to_be_visible()
    page.get_by_role("button", name="Done").click()
    expect(wizard).to_be_hidden()

    page.reload()
    dataset_row = page.locator(".dataset-row").filter(has_text=dataset_name)
    expect(dataset_row).to_have_count(1)
    expect(dataset_row.locator(".dataset-row-facts dd").nth(0)).to_have_text("2")
    expect(dataset_row.locator(".dataset-row-facts dd").nth(1)).to_have_text("6")
    dataset_row.get_by_role("link", name=f"Explore {dataset_name}").click()

    experiments = page.locator(".experiment-item")
    expect(experiments).to_have_count(2)
    experiments.nth(1).click()
    worksheet = page.get_by_role("table", name=re.compile(r"^Measurements of"))
    expect(worksheet.locator("tbody tr")).to_have_count(3)
    expect(worksheet.locator("input")).to_have_count(0)

    page.get_by_role("button", name="Edit dataset").click()
    cell = page.get_by_label("Uptake for measurement 1")
    cell.fill("0.5")
    cell.press("Enter")
    with page.expect_response(
        lambda response: response.request.method == "PATCH" and "/api/v1/datasets/" in response.url
    ) as save_response:
        page.get_by_role("button", name="Save changes").click()
    assert save_response.value.status == 200
    expect(page.get_by_text("Changes saved.", exact=True)).to_be_visible()

    page.reload()
    # Columns after the row header: source row, pressure, uptake.
    expect(worksheet.locator("tbody tr").first.locator("td").nth(2)).to_have_text("0.5")
    page.get_by_role("link", name="← Datasets").click()

    dataset_row = page.locator(".dataset-row").filter(has_text=dataset_name)
    with page.expect_response(
        lambda response: (
            response.request.method == "DELETE" and "/api/v1/datasets/" in response.url
        )
    ) as delete_response:
        dataset_row.get_by_role("button", name=f"Delete {dataset_name}").click()
        dialog = page.get_by_role("alertdialog", name=f"Delete {dataset_name}")
        expect(dialog).to_be_visible()
        dialog.get_by_role("button", name="Delete dataset").click()
    assert delete_response.value.status == 204
    expect(dataset_row).to_have_count(0)

    page.reload()
    expect(page.locator(".dataset-row").filter(has_text=dataset_name)).to_have_count(0)


###############################################################################
def test_csv_import_explore_edit_and_delete(
    page_context: Page,
) -> None:
    page = page_context
    page.goto("/datasets")
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


###############################################################################
@pytest.mark.parametrize("extension", [".xls", ".xlsx"])
def test_excel_import_persists_inspection_and_reload(
    page_context: Page,
    extension: str,
) -> None:
    page = page_context
    page.goto("/datasets")
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
