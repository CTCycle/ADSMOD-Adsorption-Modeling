# ADS-T2-05 rendered browser state

- Route: `http://127.0.0.1:5173/fitting`
- Browser: Codex in-app browser, default desktop viewport `554 × 637` pixels.
- State: after reloading the page following a completed fit, the fitting log
  showed `[INFO] Restored the last completed fitting run.` The visible result
  identified `fitting_test_82640914 · Experiment A`, `COMPLETED FIT`, best
  model `Langmuir`, `1 / 1` models, and `14` observations. The result row
  showed RMSE `2.49285`, R² `-8.44186`, and AICc `73.7062`.
- The page listed all nine model cards: Langmuir, Sips, Freundlich, Temkin,
  Toth, Dubinin–Radushkevich, Dual-Site Langmuir, Redlich–Peterson, and
  Jovanovic. Their parameter forms were inspected during the live session.
- The reload and API fetch were verified while the official launcher was
  online. The accessibility snapshot was captured after clean shutdown, so its
  backend badge reads Offline; this is teardown state, not a reload failure.
- The API E2E log records five passing live server checks. A durable browser
  console dump was not captured.
