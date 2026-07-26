import os

import uvicorn


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    offline_only = os.getenv("OFFLINE_ONLY", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    allow_private_lan = os.getenv("ALLOW_PRIVATE_LAN", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    public_internet_mode = os.getenv(
        "PUBLIC_INTERNET_MODE", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if public_internet_mode:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise SystemExit(
                "Public tunnel mode keeps the application origin on loopback; use HOST=127.0.0.1"
            )
        if offline_only:
            raise SystemExit("Public tunnel mode requires OFFLINE_ONLY=false")
        if not os.getenv("APP_API_KEY", "").strip():
            raise SystemExit("Public tunnel mode requires APP_API_KEY")
        if len(os.getenv("APP_SESSION_SECRET", "")) < 32:
            raise SystemExit(
                "Public tunnel mode requires APP_SESSION_SECRET of at least 32 characters"
            )
    elif offline_only and allow_private_lan:
        if host not in {"0.0.0.0", "::"}:
            raise SystemExit(
                "Offline LAN mode requires HOST=0.0.0.0 or ::"
            )
        if not os.getenv("APP_API_KEY", "").strip():
            raise SystemExit("Offline LAN mode requires a non-empty APP_API_KEY")
    elif offline_only and host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(
            "Local offline mode requires HOST=127.0.0.1, localhost, or ::1"
        )
    uvicorn.run(
        "clinical_app.api:app",
        host=host,
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
