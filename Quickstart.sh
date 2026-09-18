# 1. Get the repo
git clone https://github.com/sanmmie/delta-os-core.git
cd delta-os-core

# 2. Make the core importable (pick one)
export PYTHONPATH="$PWD:$PYTHONPATH"
#   or: pip install -e .

# 3. Add MarketPulse alongside it, then:
cd marketpulse
pip install -r requirements.txt

# 4. Run the demo
python -m marketpulse.cli --signals 8

# 5. Tests
pytest -q