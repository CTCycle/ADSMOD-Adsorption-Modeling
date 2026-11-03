# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from server.common.units import CustomUnit, UnitRegistry
from server.repositories.database.manager import DatabaseManager
from server.repositories.schemas.models import CustomUnitRecord

###############################################################################
class UnitRepository:

    # -------------------------------------------------------------------------
    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    # -------------------------------------------------------------------------
    @staticmethod
    def _as_dict(record: CustomUnitRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "quantity": record.quantity,
            "symbol": record.symbol,
            "factor_to_canonical": record.factor_to_canonical,
        }

    # -------------------------------------------------------------------------
    def list_units(self) -> list[dict[str, Any]]:
        statement = select(CustomUnitRecord).order_by(
            CustomUnitRecord.quantity, CustomUnitRecord.normalized_symbol
        )
        with self.database.session_factory() as session:
            return [self._as_dict(row) for row in session.scalars(statement)]

    # -------------------------------------------------------------------------
    def custom_units(self) -> list[CustomUnit]:
        return [
            CustomUnit(item["quantity"], item["symbol"], item["factor_to_canonical"])
            for item in self.list_units()
        ]

    # -------------------------------------------------------------------------
    def add(
        self, quantity: str, symbol: str, factor_to_canonical: float | None
    ) -> dict[str, Any]:
        unit = CustomUnit(quantity, symbol.strip(), factor_to_canonical)
        built_in = UnitRegistry.aliases_for(quantity).get(unit.key)
        if built_in is not None:
            raise ValueError(
                f"'{unit.symbol}' is already available as the built-in unit '{built_in}'."
            )
        with self.database.transaction() as session:
            existing = session.scalar(
                select(CustomUnitRecord).where(
                    CustomUnitRecord.quantity == quantity,
                    CustomUnitRecord.normalized_symbol == unit.key,
                )
            )
            if existing is not None:
                raise ValueError(
                    f"The {quantity} unit '{existing.symbol}' is already saved."
                )
            record = CustomUnitRecord(
                quantity=quantity,
                symbol=unit.symbol,
                normalized_symbol=unit.key,
                factor_to_canonical=factor_to_canonical,
            )
            session.add(record)
            session.flush()
            return self._as_dict(record)

    # -------------------------------------------------------------------------
    def delete(self, unit_id: int) -> None:
        with self.database.transaction() as session:
            result = session.execute(
                delete(CustomUnitRecord).where(CustomUnitRecord.id == unit_id)
            )
            if not result.rowcount:
                raise LookupError(f"Custom unit {unit_id} does not exist.")
