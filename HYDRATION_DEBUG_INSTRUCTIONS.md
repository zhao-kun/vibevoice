# Hydration Error Debugging - Testing Instructions

## Changes Made

### 1. Added Diagnostic Logging

**ProjectContext.tsx:**
- Logs when component mounts
- Logs each step of `loadProjects()` function
- Logs when projects are fetched from API
- Logs when project is restored from localStorage
- Logs when loading completes

**GenerateVoice page:**
- Logs when component mounts on client
- Logs state changes (mounted, loading, currentProject)
- Logs when showing loading state
- Logs when redirecting
- Logs before each render decision

### 2. Added Hydration Mismatch Suppression

Added `suppressHydrationWarning` to all loading/redirect div elements in:
- `frontend/app/generate-voice/page.tsx`
- `frontend/app/voice-editor/page.tsx`
- `frontend/app/speaker-role/page.tsx`

This tells React: "It's OK if server HTML differs from client render for this element"

## Testing Steps

### 1. Build Frontend

```bash
cd /home/zhaokun/work/zhao-kun/vibevoice/frontend
npm run build
```

### 2. Deploy to Backend

```bash
cp -r out/* ../backend/dist/
```

### 3. Start Backend

```bash
cd ../backend
python run.py
```

### 4. Test in Browser

1. Open browser to: `http://localhost:9527/speaker-role`
2. Select a project (any project)
3. Page should display normally
4. **Press F5 to refresh**
5. **Open Browser Console** (F12 → Console tab)
6. Look for console logs

### Expected Console Output

If working correctly, you should see:

```
[ProjectContext] Loading projects on mount
[ProjectContext] loadProjects: Starting...
[GenerateVoice] Component mounted on client
[GenerateVoice] Showing loading state: { mounted: true, loading: true }
[ProjectContext] loadProjects: Got 1 projects
[ProjectContext] loadProjects: Saved project ID from localStorage: test
[ProjectContext] loadProjects: Restored project: test
[ProjectContext] loadProjects: Complete, setting loading=false
[GenerateVoice] State check: { mounted: true, loading: false, hasProject: true }
```

### If Error Still Occurs

Please copy the **ENTIRE console output** and send it. Look for:

1. **Any error messages** (especially React error #418)
2. **The sequence of log messages** - what order they appear
3. **State values** in the logs - are they what we expect?
4. **Missing logs** - are any expected logs missing?

## What We're Looking For

The logs will tell us:

### Scenario A: Hydration Mismatch Still Happening

```
Error: Minified React error #418
[GenerateVoice] Render: { mounted: false, loading: true, hasProject: false }
[GenerateVoice] Render: { mounted: true, loading: true, hasProject: false }
[ProjectContext] Loading projects on mount
```

**This means:** React is rendering before the context is ready

### Scenario B: Race Condition

```
[ProjectContext] loadProjects: Starting...
[GenerateVoice] State check: { mounted: true, loading: false, hasProject: false }
[GenerateVoice] Redirecting to home page
[ProjectContext] loadProjects: Got 1 projects
```

**This means:** Redirect happens before projects finish loading

### Scenario C: localStorage Issue

```
[ProjectContext] loadProjects: Saved project ID from localStorage: null
[GenerateVoice] State check: { mounted: true, loading: false, hasProject: false }
[GenerateVoice] Redirecting to home page
```

**This means:** Project not saved to localStorage or cleared

## Additional Debugging

If the error persists, try these tests:

### Test 1: Check localStorage

In browser console, run:
```javascript
localStorage.getItem('vibevoice_current_project')
```

Should return project ID (e.g., "test"). If null, that's the problem.

### Test 2: Check Network

Open DevTools → Network tab → Refresh page
Look for API call to `/api/v1/projects`
Does it return your projects?

### Test 3: Test Development Mode

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000/generate-voice`
Does refresh work correctly in dev mode?

## Summary

The diagnostic logs will help us understand:
1. **When** each component renders
2. **What state** it has at each render
3. **Why** the hydration mismatch occurs
4. **Where** in the flow the issue happens

Please share the console output after testing!
