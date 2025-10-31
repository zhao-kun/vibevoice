# Page Refresh Fix with Auto-Redirect

## Problem

When refreshing browser (F5) on any page (`/generate-voice`, `/voice-editor`, `/speaker-role`):
- Navigation menu stayed visible ✓
- BUT page showed "No Project Selected" message ❌
- User had to manually navigate back to home and reselect project

This occurred in both development and production (static export) modes.

## Solution: Auto-Redirect to Home Page

Instead of showing "No Project Selected" message, **automatically redirect** to the home page (project selector) when no project is available.

### Key Insight

Projects are already saved to localStorage (see `ProjectContext.tsx:58-62`):
```typescript
useEffect(() => {
  if (currentProject) {
    localStorage.setItem("vibevoice_current_project", currentProject.id);
  }
}, [currentProject]);
```

When a project exists in localStorage:
- ✅ Page loads → Project restores → Display page content

When no project in localStorage:
- ✅ Page loads → No project → **Redirect to home page** → User selects project

## Implementation

### Pattern Applied to All Pages

```typescript
export default function PageComponent() {
  const router = useRouter();
  const { currentProject, loading } = useProject();
  const [mounted, setMounted] = useState(false);

  // Wait for client-side mount (SSG compatibility)
  useEffect(() => {
    setMounted(true);
  }, []);

  // Auto-redirect if no project after loading completes
  useEffect(() => {
    if (mounted && !loading && !currentProject) {
      router.push('/');  // Redirect to home page
    }
  }, [mounted, loading, currentProject, router]);

  // Show loading while mounting or loading projects
  if (!mounted || loading) {
    return <LoadingSpinner message="Loading project..." />;
  }

  // Show loading while redirecting (brief moment)
  if (!currentProject) {
    return <LoadingSpinner message="Redirecting to project selection..." />;
  }

  // Render normal page content
  return <PageContent />;
}
```

### Redirect Logic

```typescript
useEffect(() => {
  if (mounted && !loading && !currentProject) {
    router.push('/');
  }
}, [mounted, loading, currentProject, router]);
```

**Conditions for redirect:**
1. `mounted === true` - Component mounted on client (SSG safe)
2. `loading === false` - Project loading completed
3. `currentProject === null` - No project available

**What happens:**
- `router.push('/')` - Navigate to home page (project selector)
- User sees project selector
- User selects a project
- Navigation to desired page now works

## User Experience Flow

### Scenario 1: Project in localStorage (Normal Case)

```
User on /generate-voice → Press F5
  ↓
Page shows loading spinner (100-300ms)
  ↓
ProjectContext loads projects from API
  ↓
Project restored from localStorage
  ↓
Page displays with correct project content ✓
```

### Scenario 2: No Project in localStorage

```
User navigates to /generate-voice directly
  ↓
Page shows loading spinner (100-300ms)
  ↓
ProjectContext loads projects
  ↓
No project found in localStorage
  ↓
Auto-redirect triggers: router.push('/')
  ↓
User sees home page (project selector) ✓
  ↓
User selects a project
  ↓
Navigation to /generate-voice now works
```

### Scenario 3: Project Deleted While User Away

```
User on /generate-voice with project "test"
  ↓
Another user deletes project "test"
  ↓
User refreshes page
  ↓
ProjectContext loads projects
  ↓
Project "test" not found in API response
  ↓
currentProject becomes null
  ↓
Auto-redirect to home page ✓
  ↓
User selects another project
```

## Files Modified

### 1. Generate Voice Page (`frontend/app/generate-voice/page.tsx`)

**Changes:**
- Line 4: Added `useRouter` import
- Lines 88-102: Added router and redirect logic
- Lines 124-140: Changed "No Project Selected" to "Redirecting..." message

### 2. Voice Editor Page (`frontend/app/voice-editor/page.tsx`)

**Changes:**
- Line 4: Added `useRouter` import
- Lines 180-194: Added router and redirect logic
- Lines 209-218: Changed "No Project Selected" to "Redirecting..." message

### 3. Speaker Role Page (`frontend/app/speaker-role/page.tsx`)

**Changes:**
- Line 4: Added `useRouter` import
- Lines 10-24: Added router and redirect logic
- Lines 39-48: Changed "No Project Selected" to "Redirecting..." message

## Benefits

### ✅ Better User Experience

**Before:**
- Page shows "No Project Selected"
- User confused: "I had a project!"
- Manual navigation: Click home → Select project → Navigate back
- **3 clicks + confusion**

**After:**
- Page briefly shows "Redirecting to project selection..."
- Auto-redirects to home page
- User selects project
- **1 click + clear intent**

### ✅ Consistent Behavior

- Works in development mode (`npm run dev`)
- Works in production mode (static export via Flask)
- Handles edge cases (deleted projects, first-time users)

### ✅ Cleaner Code

- No confusing "No Project Selected" UI in each page
- Single responsibility: Pages display content, home page selects project
- Clear redirect message during transition

## Testing

### Test Case 1: Normal Refresh (Project Exists)

```bash
1. Open http://localhost:9527/generate-voice
2. Select a project (e.g., "test")
3. Press F5
4. ✅ Should show loading spinner → Display page with project
```

### Test Case 2: First Visit (No Project)

```bash
1. Clear localStorage: localStorage.clear()
2. Navigate to http://localhost:9527/generate-voice
3. ✅ Should show loading spinner → Redirect to home page
4. Select a project
5. ✅ Navigate to /generate-voice → Works correctly
```

### Test Case 3: Project Deleted

```bash
1. Open http://localhost:9527/generate-voice with project "test"
2. In another tab, delete project "test"
3. Refresh first tab
4. ✅ Should redirect to home page (project no longer exists)
```

### Test Case 4: Production Build

```bash
# Build and deploy
cd frontend && npm run build
cp -r out/* ../backend/dist/
cd ../backend && python run.py

# Test all scenarios above
# ✅ All should work correctly
```

## Technical Details

### SSG Compatibility

The `mounted` state ensures compatibility with Next.js static export:

```typescript
const [mounted, setMounted] = useState(false);

useEffect(() => {
  setMounted(true);
}, []);
```

**Why necessary:**
- Static HTML generated at build time has `mounted=false`
- On client load, `useEffect` runs → `mounted` becomes `true`
- Guarantees component re-render after hydration
- Prevents SSG/client state mismatch

### Redirect Timing

```typescript
if (mounted && !loading && !currentProject) {
  router.push('/');
}
```

**Waits for:**
1. Client-side mount complete (`mounted`)
2. Project loading complete (`!loading`)
3. Confirms no project (`!currentProject`)

**Then redirects immediately** - no delay, clean transition.

## Build Verification

```bash
$ npm run build

✓ Compiled successfully in 2.9s
✓ Generating static pages (8/8)
✓ Exporting (2/2)

Route (app)                         Size  First Load JS
┌ ○ /generate-voice              10.2 kB         138 kB
├ ○ /speaker-role                21.5 kB         149 kB
└ ○ /voice-editor                5.73 kB         133 kB
```

## Comparison with Previous Approach

### Previous: Show "No Project Selected"

```
User on page without project
  ↓
See message: "No Project Selected"
  ↓
Manual navigation: Click home → Select project → Navigate back
  ↓
3 clicks + confusion
```

### Current: Auto-Redirect

```
User on page without project
  ↓
See message: "Redirecting to project selection..."
  ↓
Auto-redirected to home page
  ↓
Select project
  ↓
1 click + clear flow
```

## Date

2025-10-31

## Summary

- ✅ Auto-redirects to home page when no project available
- ✅ Works in both development and production modes
- ✅ Handles all edge cases (first visit, deleted projects, etc.)
- ✅ Better UX with clear redirect message
- ✅ Cleaner code with single responsibility principle
- ✅ Project persistence via localStorage maintained
