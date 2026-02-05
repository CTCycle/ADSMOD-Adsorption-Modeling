from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from ADSMOD.server.utils.constants import (
    DOCS_ENDPOINT,
    FASTAPI_DESCRIPTION,
    FASTAPI_TITLE,
    FASTAPI_VERSION,
    ROOT_ENDPOINT,
)
from ADSMOD.server.routes.datasets import router as dataset_router
from ADSMOD.server.routes.fitting import router as fit_router
from ADSMOD.server.routes.browser import router as browser_router
from ADSMOD.server.routes.training import router as training_router
from ADSMOD.server.routes.nist import router as nist_router


###############################################################################
app = FastAPI(
    title=FASTAPI_TITLE,
    version=FASTAPI_VERSION,
    description=FASTAPI_DESCRIPTION,
)

app.include_router(dataset_router)
app.include_router(fit_router)
app.include_router(browser_router)
app.include_router(training_router)
app.include_router(nist_router)


# -------------------------------------------------------------------------
@app.get(ROOT_ENDPOINT)
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url=DOCS_ENDPOINT)
