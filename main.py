"""Совместимая точка входа: весь симулятор находится в пакете akim_sim.

python main.py --input example_allocation.json --offline --json
"""

from akim_sim.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
