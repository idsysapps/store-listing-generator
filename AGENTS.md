# Store Listing Generator - Project Standards

## Overview

This document establishes coding standards and practices for the Store Listing Generator project. All code changes should follow these guidelines.

---

## Testing Requirements

### Test-Driven Development (TDD)

**All new functionality, bug fixes, and implementations MUST follow the TDD cycle:**

1. **RED**: Write a failing test first that describes expected behavior
2. **GREEN**: Write minimum code to make the test pass
3. **REFACTOR**: Clean up code while keeping tests green

### Testing Philosophy

- **Black-box testing**: Test behavior (inputs → outputs), not implementation details
- **Inputs**: props, user events, data streams
- **Outputs**: rendered DOM, emitted events, side effects

### Anti-Patterns (Never Do These)

```javascript
// ❌ NEVER: Test internal state
expect(wrapper.vm.count).toBe(1)

// ❌ NEVER: Call internal methods
wrapper.vm.increment()

// ❌ NEVER: Arbitrary sleeps
Thread.sleep(5000)
await new Promise(r => setTimeout(r, 5000))

// ❌ NEVER: CSS class selectors
expect(wrapper.find('.error-text').exists()).toBe(true)

// ❌ NEVER: Snapshot entire component trees
expect(wrapper.html()).toMatchSnapshot()
```

---

## Frontend Standards

### Vue Component Testability Requirements

**Every Vue component MUST provide:**

#### 1. `data-testid` Attributes

Use on all interactive elements:

```vue
<Button data-testid="btn-submit" @click="handleSubmit" />
<div data-testid="form-container" />
<input data-testid="input-name" />
```

For PrimeVue components, use **Pass Through (pt)** API:

```vue
<FileUpload
  :pt="{
    chooseButton: { 'data-testid': 'file-select-button' },
    input: { 'data-testid': 'file-input' }
  }"
/>
```

#### 2. Expose State via `defineExpose`

```javascript
defineExpose({
  getState,      // () => { isLoading, isReady, data, error }
  getFormValues, // () => formData object
  validate,      // () => boolean
  reset         // () => void
})
```

#### 3. Emit Custom Events for Async Operations

```javascript
const emit = defineEmits(['submitStart', 'submitSuccess', 'submitError'])

async function handleSubmit() {
  emit('submitStart')
  try {
    const result = await api.save(data)
    emit('submitSuccess', result)
  } catch (err) {
    emit('submitError', err)
  }
}
```

#### 4. Window Global for E2E Testing

```javascript
if (typeof window !== 'undefined') {
  window.__componentRef = {
    getState: () => ({ isReady, isSubmitting, submitResult }),
    waitForReady: (timeout = 5000) => { /* promise-based */ },
    waitForSubmitResult: (timeout = 30000) => { /* promise-based */ }
  }
}
```

### Pinia Store Pattern (Elm Architecture)

Use TEA pattern for testable stores:

```javascript
// stores/todos-update.js - Pure function, framework-agnostic
export function update(model, message) {
  switch (message.type) {
    case 'ADD_TODO':
      return { ...model, todos: [...model.todos, message.todo] }
  }
}

// stores/todos.js - Public store
const privateStore = defineStore('private-todos', () => {
  const model = ref({ todos: [] })
  return { model }
})

export const useTodosStore = defineStore('todos', () => {
  const privateStore = usePrivateStore()

  function dispatch(message) {
    privateStore.model = update(privateStore.model, message)
  }

  return {
    todos: readonly(privateStore.model.todos),
    dispatch
  }
})
```

**Test the pure function directly:**
```javascript
import { update } from './todos-update'
describe('update', () => {
  it('adds a todo', () => {
    const result = update({ todos: [] }, { type: 'ADD_TODO', todo: { id: 1, text: 'Test' } })
    expect(result.todos).toHaveLength(1)
  })
})
```

---

## Backend Standards

### Service Layer Testing

- Keep business logic in services (not resources)
- Use interfaces for dependencies (enables mocking)
- Test service methods directly

```java
// GOOD - testable
@Inject
EstimateService estimateService;

@Test
void shouldCalculatePrice() {
    when(priceCalculator.calculate(any())).thenReturn(100.00);
    Price result = estimateService.calculatePrice(estimate);
    assertThat(result).isEqualTo(100.00);
}
```

### Test File Naming

| Type | Pattern | Example |
|------|---------|---------|
| Unit Test | `*Test.java` | `EstimateServiceTest.java` |
| Integration Test | `*IT.java` | `GangSheetResourceIT.java` |
| E2E Test | `*E2ETest.java` | `EstimateViewE2ETest.java` |
| Page Object | `*Page.java` | `EstimateFormPage.java` |

---

## E2E Testing Standards

### Page Objects

Every page/dialog should have a Page Object:

```java
public class GangSheetFormDialog {
    private WebDriver driver;

    // Use data-testid when available
    private static final By SUBMIT_BTN = By.cssSelector("[data-testid='btn-submit']");
    private static final By DIALOG = By.cssSelector(".p-dialog");

    public void waitForDialog() {
        // Wait for component state, not arbitrary sleep
        waitForDialogWithHook(10);
    }
}
```

### Wait for State, Not Time

```java
// ❌ BAD
Thread.sleep(5000);

// ✅ GOOD - Use component hooks
waitForSubmitResult(30000);

// ✅ BEST - Use promise-based hooks
js.executeScript("return window.__formRef.waitForSubmitResult()");
```

### Critical: Use JavaScript Click for Vue/React

**Known issue:** Selenium's `element.click()` may not trigger Vue/React events (Vue attaches listeners asynchronously after DOM render).

```java
// ❌ BAD - Native click may not work
driver.findElement(BTN).click();

// ✅ GOOD - JavaScript click bypasses this issue
WebElement element = waitForClickable(locator);
((JavascriptExecutor) driver).executeScript("arguments[0].click();", element);
```

---

## Running Tests

```bash
# Frontend unit tests
cd frontend && npm run test

# Backend unit tests
cd backend && mvn test

# Backend integration tests
cd backend && mvn verify

# E2E tests (requires infrastructure)
make test-e2e
```

---

## Test Data

### E2E Test Users

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@example.com | admin123 |
| Internal | internal@example.com | internal123 |
| Customer | customer@example.com | customer123 |

---

## Additional Documentation

- `TESTING_GUIDELINES.md` - Comprehensive testing documentation
- `TESTING.md` - Testing workflow and categories
- `frontend/src/components/gangsheets/GangSheetForm.vue` - Example of testable component

---

## Git Workflow

### Feature Branch Process

**All changes MUST be done via feature branches and PRs, not direct pushes to master:**

1. **Before starting work, create or identify the issue:**
   - Every piece of work MUST have a corresponding GitHub issue
   - If no issue exists, create one first
   - The issue will be closed by your PR

2. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   # OR for bug fixes:
   git checkout -b fix/your-bug-fix
   ```

3. **Make changes following TDD:**
   - RED: Write failing tests first
   - GREEN: Implement the feature
   - REFACTOR: Clean up

4. **Commit and push:**
   ```bash
   git add -A && git commit -m "feat: description (closes #issue-number)"
   git push -u origin feat/your-feature-name
   ```

5. **Create PR via GitHub CLI:**
   ```bash
   gh pr create --title "feat: your feature" --body "Closes #issue-number"
   ```

6. **After PR merged, delete the branch:**
   ```bash
   git checkout master && git pull && git branch -d feat/your-feature-name
   ```

### Branch Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feat/` | `feat/embroidery-design-tracking` |
| Bug Fix | `fix/` | `fix/login-redirect-issue` |
| Chore | `chore/` | `chore/update-dependencies` |
| Documentation | `docs/` | `docs/api-documentation` |

### Commit Message Format

Use conventional commits:
- `feat: ...` - New feature
- `fix: ...` - Bug fix
- `refactor: ...` - Code refactoring
- `test: ...` - Adding tests
- `docs: ...` - Documentation changes
- `ci: ...` - CI/CD changes

**Include issue number in commit body:** `(closes #123)`

---

## CI/CD Standards

### GitHub Actions Workflow

All CI workflows live in `.github/workflows/`:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: ['**']
  pull_request:
    branches: ['**']
  push:
    branches: ['master']
```

### Pipeline Jobs

| Job | Trigger | Purpose |
|-----|--------|---------|
| `lint-and-test` | All pushes/PRs | Ruff, Pyright, pytest |
| `build` | Master push only | Docker image build (no push) |

### Container Images

- Python: `ubi9/python:3.13`
- PostgreSQL: `registry.access.redhat.com/ubi9/postgresql-16`
- Redis: `registry.access.redhat.com/ubi9/redis-7`

### Local Master Branch Protection

A pre-commit hook prevents direct commits to master:

```bash
#!/bin/bash
if [ "$(git branch --show-current)" = "master" ]; then
    echo "❌ Committing directly to master is not allowed."
    exit 1
fi
```

---

## Pre-commit Hooks

### Installation

```bash
uv pip install pre-commit
pre-commit install
```

### Running Manually

```bash
pre-commit run --all-files
pre-commit run ruff --files src/
```

### Hooks

| Hook | Purpose |
|------|---------|
| `ruff` | Lint + format |
| `pyright` | Type check |
| `pytest` | Run tests |

---

## Docker Compose (Local Development)

### Commands

```bash
# Start services
podman-compose up -d

# Stop services
podman-compose down

# View logs
podman-compose logs -f

# Rebuild services
podman-compose up -d --build
```

### Services

| Service | Port | Purpose |
|---------|------|---------|
| `postgres` | 5432 | Database |
| `redis` | 6379 | Celery broker |
| `app` | - | Application container |

### Environment

Set via `docker-compose.yml` or `.env`:
- `DATABASE_HOST`, `DATABASE_PORT`, `DATABASE_NAME`
- `REDIS_URL`

---

## GitHub CLI

Use `gh` for all GitHub interactions:

```bash
gh issue view 1 --repo owner/repo
gh issue create --title "feat: ..." --body "..."
gh pr create --title "feat: ..." --body "Closes #123"
gh pr checkout 123
```

---

## Checklist Before Submitting Code

- [ ] Tests written first (TDD)
- [ ] New components expose `getState()` via `defineExpose`
- [ ] Interactive elements have `data-testid`
- [ ] Async operations emit `submitStart`, `submitSuccess`, `submitError`
- [ ] E2E hooks registered on `window.__*Ref`
- [ ] No `Thread.sleep()` in tests
- [ ] No testing internal state/methods
- [ ] Pinia stores follow TEA pattern
- [ ] Changes made via feature branch, not direct push to master
- [ ] GitHub issue exists for all committed work
- [ ] Commit message includes `Closes #issue-number`

---

## Python Project Standards

### Project Structure

```
store-listing-generator/
├── src/
│   └── store_listing/
│       ├── ingest/           # Trend scrapers (pytrends, requests)
│       ├── llm/               # vLLM/Ollama integration
│       ├── ip_clearance/      # USPTO, RapidFuzz checks
│       ├── rendering/         # Pillow/ImageMagick
│       ├── publishing/        # SP-API, Shopify, Etsy
│       └── orchestration/     # Celery tasks
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/              # Test data fixtures
├── pyproject.toml
└── uv.lock
```

### Python Testing Standards

| Type | Pattern | Example |
|------|---------|---------|
| Unit Test | `test_*.py` | `test_uspsto_client.py` |
| Integration Test | `*_integration_test.py` | `test_sp_api_integration_test.py` |
| E2E Test | `test_*_e2e.py` | `test_listing_pipeline_e2e.py` |
| Page Object | `*_page.py` | `approval_dashboard_page.py` |

### TDD Red/Green/Refactor Cycle (Python)

```python
# RED: Write failing test first
async def test_uspsto_search_returns_trademarks():
    client = USPTOClient()
    with pytest.fail("Not implemented: USPTO search"):
        result = await client.search_class_025("test slogan")

# GREEN: Minimum code to pass
class USPTOClient:
    async def search_class_025(self, query: str) -> list[Trademark]:
        return []

# REFACTOR: Clean up (types, error handling, docs)
```

### Python Anti-Patterns

```python
# ❌ NEVER: Test internal state
assert self._cache == {}

# ❌ NEVER: Arbitrary sleeps
time.sleep(5)

# ❌ NEVER: CSS selectors in tests
page.locator(".error-text")

# ❌ NEVER: Mock implementation details
with patch("store_listing.llm.generate"):

# ❌ NEVER: Test without type hints on public methods
def generate_slogan(trend_data):  # Missing types!
    pass
```

### Python E2E Testing (Playwright)

Every page/dialog should have a Page Object:

```python
# tests/e2e/approval_dashboard_page.py
from playwright.async_api import Page

class ApprovalDashboardPage:
    APPROVE_BTN = "[data-testid='btn-approve']"
    
    def __init__(self, page: Page):
        self.page = page
    
    async def approve_slogan(self, slogan_id: str) -> None:
        await self.page.wait_for_selector(f"[data-testid='slogan-{slogan_id}']")
        await self.page.click(self.APPROVE_BTN)
```

Wait for state, not time:

```python
# ❌ BAD
await asyncio.sleep(5)

# ✅ GOOD - Use Playwright's auto-waiting
await self.page.wait_for_selector("[data-testid='btn-approve']", state="visible")
```

### Python Tools

```bash
uv sync              # Install dependencies
uv pytest            # Run unit tests
uv pytest tests/e2e/ # Run E2E tests
uv ruff check         # Lint
uv pyright            # Type check
```

### Python Conventions

- **Type hints required** on all public function signatures
- **Pydantic models** for API/request/response validation
- **async/await** for I/O-bound operations (httpx, aiohttp)
- **pytest** + pytest-asyncio for testing
- **ruff** for linting (configured in pyproject.toml)
- **pyright** for static type checking

### Checklist Before Submitting Python Code

- [ ] Tests written first (TDD Red/Green/Refactor)
- [ ] Type hints on all public function signatures
- [ ] Pydantic models for data validation
- [ ] No `time.sleep()` in tests
- [ ] No testing internal state
- [ ] Page Objects for E2E tests
- [ ] `data-testid` attributes on interactive elements
- [ ] Pre-commit hooks pass
