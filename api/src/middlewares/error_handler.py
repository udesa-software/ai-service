from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


class MissingBiographyError(AppError):
    def __init__(self):
        super().__init__(
            status_code=400,
            message="User biography is required for recommendations",
        )


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message},
    )


async def missing_biography_handler(_request: Request, _exc: MissingBiographyError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "code": "MISSING_BIOGRAPHY",
            "error": "User biography is required for recommendations",
        },
    )
