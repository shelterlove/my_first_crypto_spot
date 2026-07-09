# Freqtrade Userdir

This directory is legacy support material for the old thin Freqtrade shell.

Current V4.8 deployment work uses the native Binance executor, not Freqtrade.
Only `strategies/CryptoSpotV26.py` is retained because local adapter tests
still exercise the base shell helpers.

Do not add new Freqtrade shells for V4.8 unless there is a separate design to
reproduce the execution-transform and spot cash-constraint behavior exactly.
