# 📘 Project Best Practices

## 1. Project Purpose
Vehicle Management System providing:
- Registration and management of vehicles and owners
- QR-based I-Card generation for vehicles
- Real-time queueing of vehicles for dispatch with rank-based release
- Push notifications (FCM) to active devices for queue updates
- Reporting and exports of queue history

Primary domain: FastAPI-based backend with Google Firestore as the data store and Firebase Admin for messaging.

## 2. Project Structure
- main.py
  - FastAPI application factory: sets CORS and includes versioned routers under /v1
- app/
  - config/
    - config.py: static configuration constants (JWT, Firestore collection names, service account path)
  - routers/
    - authentication.py: signup/login, JWT issuance
    - vehicle_crud.py: vehicle registration, update, delete, fetching, and I-Card download
    - queue_management.py: verify QR, add/release vehicles from queue, FCM-related endpoints
    - reporting.py: export queue history report to Excel
  - services/
    - firebase.py: Firebase Admin initialization and Firestore client
    - firestore_utils.py: CRUD utilities for vehicles
    - login.py: JWT token creation
    - pydantic.py: request/response models and enums (Pydantic v2)
    - pdf_generator.py: I-Card PDF generation and QR code composition
    - vehicle_queue_utils.py: queue management helpers and history logging
  - image utils/
    - i_card_logo.png (logo used in PDFs)

Observations and guidance:
- Separation of concerns is reasonable: routers expose HTTP endpoints; services/utilities interact with Firestore and other infra concerns.
- Use a consistent directory naming convention (avoid spaces). Prefer app/image_utils/.
- Versioning is in place via app.include_router(..., prefix="/v1"). Retain this and evolve via new prefixes (/v2).
- Centralize shared constants (collections, field names) in config to avoid drift.

## 3. Test Strategy
Current state:
- No formal tests present; only a test.ipynb. Establish a proper pytest-based test suite.

Recommendations:
- Framework: pytest + httpx (or FastAPI TestClient) for API tests.
- Structure:
  - tests/
    - unit/ (services, utils)
    - integration/ (routers via app instance)
    - e2e/ (optional, with emulator or staging)
- Firestore: Use the Google Cloud Firestore Emulator for integration tests. Parameterize the DB client so tests can point to emulator.
- Mocking:
  - For unit tests, monkeypatch or dependency-inject the Firestore client and Firebase messaging (e.g., patch db and messaging.send).
  - Prefer dependency injection via FastAPI Depends or a simple repository pattern to enable swapping implementations.
- Naming: test_<module>_<behavior>.py; functions test_<unit_under_test>_<expected_outcome>.
- Coverage: Target >85% line coverage for services/utils, >70% for routers.
- Types: Add mypy to catch type issues around async/sync boundaries and Firestore payload shapes.

## 4. Code Style
- General
  - Python 3.10+ recommended.
  - Use Black (formatting), isort (imports), and Ruff/Flake8 (linting). Add a pre-commit hook.
  - Prefer type hints across all modules. Check with mypy.
- Async I/O
  - FastAPI endpoints can be async, but Firestore Admin SDK calls are synchronous. Avoid blocking the event loop.
    - For heavy Firestore operations inside async endpoints, offload to a thread: anyio.to_thread.run_sync or asyncio.get_event_loop().run_in_executor.
  - Keep consistency: use async def for I/O-bound endpoints, def for pure CPU-light handlers.
- Pydantic (v2)
  - Keep request/response models in app/services/pydantic.py.
  - Use Enums for constrained string fields (e.g., VehicleTypeEnum, VehicleShiftEnum) as done.
  - Consider response_model for all endpoints to enforce output shape.
- Naming conventions
  - Python code: snake_case for variables/functions; PascalCase for classes.
  - Firestore documents: choose and standardize one style for field names. Recommendation: camelCase to match current vehicleDetails (e.g., registrationNumber, vehicleType, vehicleShift) and always use that style consistently across collections.
  - Provide mapping helpers when needed to convert between Python snake_case and Firestore camelCase.
- Error handling
  - Raise HTTPException for client errors (400/401/403/404) with clear messages.
  - Log exceptions with traceback for server errors; return generic 500 to clients.
  - Add global exception handlers (app.exception_handler) for ValidationError and generic Exception.
- Logging
  - Use structured logging (include request id, user, action where available).
  - Avoid logging secrets or PII (e.g., password hashes).
- Security
  - Do not hardcode secrets. Load JWT SECRET_KEY, ALGORITHM, and Firebase credentials path from environment variables or a .env file (via pydantic-settings).
  - JWTs should include exp, iat, and sub claims; add expiry and refresh as needed.
  - Hash passwords server-side using passlib (bcrypt/argon2). Do not trust client-provided passwordHash.
  - Restrict CORS in production to known origins.
  - Implement RBAC via FastAPI dependencies (e.g., get_current_user with role checks).

## 5. Common Patterns
- Routers per domain (authentication, vehicles, queue, reporting) mounted under /v1.
- Services layer wraps Firestore operations and external integrations (Firebase Admin, ReportLab, etc.).
- Pydantic models define request schemas and enums to validate input.
- Queue design:
  - vehicleQueue uses vehicle_id as document id (enforces uniqueness in queue per vehicle).
  - queue_rank determines ordering; updates on release decrement ranks of following vehicles.
  - History stored in vehicleQueueHistory with timestamps and optional check-in/out times.
- Notifications
  - Messaging via Firebase Admin; consider using topics per vehicle_type/shift instead of per-token sends for scalability.

## 6. Do's and Don'ts
- Do's
  - Use centralized constants for collection names and field keys (and reference them everywhere).
  - Validate all client input via Pydantic models and Enums.
  - Use Firestore transactions/batches when mutating multiple documents to avoid race conditions (e.g., rank updates on release/add).
  - Store UTC timestamps with timezone (ISO 8601, e.g., datetime.now(timezone.utc)).
  - Offload blocking Firestore operations from async endpoints to avoid blocking the event loop.
  - Add response_model to endpoints and document routes with tags and summaries.
  - Keep router prefixes versioned; add new versions for breaking changes.
  - Keep image and static assets under directories without spaces (e.g., app/image_utils/).
- Don'ts
  - Do not hardcode secrets or absolute OS paths (e.g., E:/...). Make them configurable.
  - Do not mix camelCase and snake_case for the same Firestore collection fields; pick one and enforce it.
  - Do not perform multiple dependent writes without a transaction when correctness matters (queue rank logic).
  - Do not store or log plaintext passwords or client-provided hashes; always hash server-side.
  - Do not allow broad CORS in production.

## 7. Tools & Dependencies
- Key libraries
  - FastAPI/Starlette: API framework
  - firebase-admin, google-cloud-firestore: Firestore access and FCM
  - jose: JWT
  - reportlab, Pillow, qrcode: PDF and QR generation
  - pandas, openpyxl: Reporting to Excel
  - uvicorn: ASGI server
- Setup
  - Python virtual environment recommended
  - pip install -r requirements.txt
  - Configuration via environment variables or .env (preferred):
    - SECRET_KEY, ALGORITHM
    - FIREBASE_CREDENTIALS_PATH (or use GOOGLE_APPLICATION_CREDENTIALS)
    - COLLECTION names if environment-specific (optional)
  - Run: uvicorn main:app --reload

## 8. Other Notes
- Domain rules
  - Only the vehicle at rank 1 can be released from the queue (enforced in queue_management.py). Document and preserve this rule.
- Known inconsistencies to fix (and keep consistent going forward)
  - activeDevices field naming mismatch: register_device stores fcmToken/isActive; token fetch expects fcm_token/is_active. Standardize on camelCase (fcmToken, isActive) or snake_case consistently, and update all call sites and cleanup logic accordingly.
  - Queue/history field naming: ensure vehicleShift is always written and read using the same case across collections.
  - login.py duplicates SECRET_KEY/ALGORITHM assignment; use config only. Also add token expiry.
  - qr_data parsing checks for "_" but splits by "__". Validate using the exact delimiter and structure.
  - Mixed async/sync: wrap Firestore calls inside async endpoints using thread offloading.
  - main.py tag name has a leading space in " Vehicles"; remove spaces in tag names.
  - app/image utils/ directory has a space; rename to app/image_utils/ and update references.
  - pdf_generator logo_path is absolute; make it configurable and relative to project or environment.
- Firestore data modeling
  - Consider using subcollections or composite indexes for frequent queries (e.g., vehicle_type + queue_rank, vehicle_type + vehicleShift).
  - Use document IDs deliberately (vehicle_id for queue items) to enforce uniqueness where intended.
- LLM generation tips
  - Always import and reuse constants from app.config.config.
  - Preserve API shapes (field names and casing) when creating/updating documents to avoid breaking clients.
  - Prefer adding service functions over writing Firestore calls directly in routers.
  - Use logging and robust exception handling; return HTTPException with clear messages for client errors.
