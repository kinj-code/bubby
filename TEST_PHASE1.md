# Phase 1: OverlayWindow Test Instructions

## Prerequisites

- Virtual environment activated: `source venv/bin/activate`
- Dependencies installed (verified above)

## Running the Test

### Option 1: Quick Interactive Test (Recommended for First Run)

```bash
# Activate virtual environment
source venv/bin/activate

# Run the interactive test
python3 test_overlay.py
```

This will:
1. Show a 400x400 transparent window for 3 seconds (Test 1)
2. Run automated click-through tests (Test 2)
3. Show an interactive window for drag-and-drop testing (Test 3)

**Keep the final window open** and test:
- Drag the window around
- Try dropping it in the bottom-right corner (red X should appear)
- Watch the terminal logs for event feedback
- Close with Alt+F4 or Ctrl+C

### Option 2: Direct Module Test

```bash
# Activate virtual environment
source venv/bin/activate

# Run the overlay module directly
python3 -m src.ui.overlay
```

This launches just the overlay window with logging enabled.

## Expected Behavior

### Visual Checks
- [ ] A 400x400 window appears (may look like a subtle square)
- [ ] No title bar or window decorations
- [ ] Window stays on top of other windows
- [ ] Background is transparent (not black/white)
- [ ] Red X button appears in bottom-right when hovering

### Interaction Checks
- [ ] Can click and drag the window
- [ ] Window follows mouse during drag
- [ ] Dropping in bottom-right corner triggers close (check logs)
- [ ] Terminal shows logs like:
  - `Mouse press at: QPoint(x, y)`
  - `Drag started`
  - `Click at: QPoint(x, y), in_close_zone=True/False`
  - `Dropped in close zone - closing`

### Click-Through Test (Test 2)
- [ ] Logs show "Click-through enabled"
- [ ] Logs show "Click-through disabled"
- [ ] No errors or assertion failures

## Troubleshooting

### Window appears with black/white background
- This is normal for some Wayland compositors
- The window is still transparent to the compositor
- Check if you can see desktop/icons through it

### Window doesn't stay on top
- Zorin OS Wayland may have different behavior
- Check compositor settings for "always on top" support

### No mouse events registered
- Ensure click_through=False in test
- Check that window has focus
- Try clicking directly on the window area

### PySide6 import errors
- Verify virtual environment is activated: `which python3`
- Should point to `/home/kinj/.continue/bubby/venv/bin/python3`
- Reinstall if needed: `pip install PySide6`

## Reporting Results

Please provide:
1. **Screenshot** of the window (if possible)
2. **Terminal output** from the test run
3. **Behavior observations**:
   - Did the window appear?
   - Was it transparent?
   - Did drag work?
   - Did close zone work?
   - Any errors in logs?

4. **Wayland-specific notes**:
   - Zorin OS version
   - Any visual artifacts
   - Compositor being used (Mutter, etc.)

## Next Steps After Successful Test

Once you confirm the OverlayWindow works correctly, we'll proceed to:
- Wayland screen capture foundation (Phase 1, Part 2)
- Basic animation engine stub (Phase 1, Part 3)

**Ready to test? Run: `python3 test_overlay.py`**