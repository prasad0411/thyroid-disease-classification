# Frontend

React 19 and strict TypeScript client for the thyroid panel review service.

```bash
npm ci
npm run dev      # http://localhost:5173, proxies /api to FastAPI on :8000
npm test         # Vitest + React Testing Library
npm run build    # type check and production bundle
```

State lives in a Context + useReducer store (`src/state`). The typed API client (`src/api.ts`) handles timeouts, aborts, FastAPI 422 details, and gateway errors. Reference intervals and SHAP labelling live in `src/reference.ts`; the copy to note text is built in `src/note.ts`.
